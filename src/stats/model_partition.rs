//! Model-level diagnostic stream partition; reusable events, background numeric output.
use std::{cell::RefCell,sync::{Arc,Mutex,OnceLock},time::{Instant,Duration}};
use candle_core::{Tensor,cuda_backend::cudarc::driver::{CudaEvent,sys}};
const PHASES:[&str;18]=["layer_start","attn_norm_end","mix_down","mix_act","mix_up","attn_mix_end","mixer_start","mixer_end","mlp_norm_end","mix_down","mix_act","mix_up","mlp_mix_end","moe_start","routed_end","shared_end","moe_end","layer_end"];
struct Layer { layer:usize,rows:usize,count:usize,events:Vec<CudaEvent> }
struct Row {layer:usize,rows:usize,ms:[f32;19]}
struct Buffer {rows:Vec<Row>,last:Instant,dropped:u64}
static BUFFER:OnceLock<Arc<Mutex<Buffer>>>=OnceLock::new();
thread_local! {static PENDING:RefCell<Vec<Layer>>=const{RefCell::new(Vec::new())};static FREE:RefCell<Vec<Layer>>=const{RefCell::new(Vec::new())};static ACTIVE:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}
pub(crate) fn install(){if std::env::var_os("NAIWA_MODEL_PARTITION_FILE").is_some()||super::head_partition::enabled(){assert!(mistralrs_core::qwen4_split::set_boundary_probe(dispatch));}}
fn dispatch(xs:&Tensor,layer:usize,phase:&'static str){
 if phase.starts_with("sample_")||phase.starts_with("pipeline_"){super::head_partition::host_mark(phase);return;}
 if matches!(phase,"tail_start"|"head_start"|"head_end"|"next_model"){super::head_partition::mark(xs,phase);return;}
 if std::env::var_os("NAIWA_MODEL_PARTITION_FILE").is_some(){mark(xs,layer,phase);}
}
fn buffer()->&'static Arc<Mutex<Buffer>>{BUFFER.get_or_init(||{
 let b=Arc::new(Mutex::new(Buffer{rows:Vec::with_capacity(65536),last:Instant::now(),dropped:0}));let writer=b.clone();
 let file=std::fs::File::create(std::env::var_os("NAIWA_MODEL_PARTITION_FILE").unwrap()).expect("model partition file");
 std::thread::spawn(move||{use std::io::Write;let mut f=std::io::BufWriter::new(file);loop{std::thread::sleep(Duration::from_millis(100));let batch={let mut b=writer.lock().unwrap();if b.rows.is_empty()||b.last.elapsed()<Duration::from_millis(300){continue}let rows=std::mem::replace(&mut b.rows,Vec::with_capacity(65536));let drops=b.dropped;b.dropped=0;(rows,drops)};for r in batch.0{let mut ep:Vec<_>=PHASES.iter().zip(r.ms).enumerate().map(|(i,(p,t))|{let name=match i {2=>"attn_down",3=>"attn_act",4=>"attn_up",9=>"mlp_down",10=>"mlp_act",11=>"mlp_up",_=>p};serde_json::json!([name,t])}).collect();ep.push(serde_json::json!(["next_layer",r.ms[18]]));writeln!(f,"{}",serde_json::json!({"layer":r.layer,"rows":r.rows,"endpoints_ms":ep})).unwrap();}writeln!(f,"{}",serde_json::json!({"flush":true,"dropped":batch.1})).unwrap();f.flush().unwrap();}});b
})}
fn mark(xs:&Tensor,layer:usize,phase:&'static str){
 static STRIDE:OnceLock<usize>=OnceLock::new();let stride=*STRIDE.get_or_init(||std::env::var("NAIWA_MODEL_PARTITION_STRIDE").ok().and_then(|v|v.parse().ok()).unwrap_or(1).max(1));
 if phase=="layer_start"{ACTIVE.with(|a|a.set(layer%stride==0));}
 if !ACTIVE.with(|a|a.get()){return;}
 if !xs.device().is_cuda(){return}let stream=xs.device().as_cuda_device().unwrap().cuda_stream();let b=buffer();
 PENDING.with(|p|{let mut p=p.borrow_mut();
 if phase=="layer_start"{let mut a=FREE.with(|f|f.borrow_mut().pop()).unwrap_or_else(||Layer{layer,rows:0,count:0,events:(0..18).map(|_|stream.context().new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap()).collect()});a.layer=layer;a.rows=0; // Set from actual mixer input shape below.
 a.count=0;p.push(a);}
 let a=p.last_mut().expect("model layer");
 // Final head's HC mix is outside an active decoder layer.
 if a.count==18 && phase.starts_with("mix_"){return;}
 if phase=="mixer_start"{a.rows=xs.elem_count()/xs.dims().last().copied().unwrap_or(1);}
 assert_eq!(PHASES[a.count],phase);a.events[a.count].record(&stream).unwrap();a.count+=1;
 // Routed host reads in previous two layers completed before reaching layer_start here.
 if phase=="layer_start"&&p.len()>2{let a=&p[0];assert_eq!(a.count,18);let mut ms=[0.;19];for (i,e) in a.events.iter().enumerate(){ms[i]=a.events[0].elapsed_ms(e).unwrap();}ms[18]=a.events[0].elapsed_ms(&p[1].events[0]).unwrap();let mut b=b.lock().unwrap();if b.rows.len()<65536{b.rows.push(Row{layer:a.layer,rows:a.rows,ms});}else{b.dropped+=1;}b.last=Instant::now();drop(b);let old=p.remove(0);FREE.with(|f|f.borrow_mut().push(old));}
 });
}
