//! Opt-in CUDA-event diagnostic with reusable events and idle background formatting.
//! Events span host submission gaps as well as kernels. No added stream synchronize.
use std::{cell::RefCell, sync::{Arc,Mutex,OnceLock},time::{Duration,Instant}};
use candle_core::cuda_backend::cudarc::driver::{CudaEvent,CudaStream,sys};
const PHASES:[&str;9]=["route","plan_submit","preload_ready","resident_end","staging_ready","staged_end","promotion_end","cpu_ready","combine_end"];
struct Layer { layer:usize, rows:usize, count:usize, events:Vec<CudaEvent> }
struct Row { layer:usize,rows:usize,ms:[f32;10] }
struct Buffer { rows:Vec<Row>,last:Instant,dropped:u64 }
static BUFFER:OnceLock<Arc<Mutex<Buffer>>>=OnceLock::new();
thread_local! { static PENDING:RefCell<Vec<Layer>>=const{RefCell::new(Vec::new())}; static FREE:RefCell<Vec<Layer>>=const{RefCell::new(Vec::new())}; }
#[inline]
pub(crate) fn enabled()->bool { crate::config::diag().gpu_partition_file.is_some() }
fn buffer()->&'static Arc<Mutex<Buffer>> { BUFFER.get_or_init(|| {
 let buffer=Arc::new(Mutex::new(Buffer{rows:Vec::with_capacity(65536),last:Instant::now(),dropped:0}));let writer=buffer.clone();
 let path=crate::config::diag().gpu_partition_file.as_ref().unwrap();
 let file=std::fs::File::create(path).expect("partition file");
 std::thread::spawn(move||{ use std::io::Write;let mut f=std::io::BufWriter::new(file);loop{std::thread::sleep(Duration::from_millis(100));let batch={let mut b=writer.lock().unwrap();if b.rows.is_empty()||b.last.elapsed()<Duration::from_millis(300){continue}let rows=std::mem::replace(&mut b.rows,Vec::with_capacity(65536));let dropped=b.dropped;b.dropped=0;(rows,dropped)};for r in batch.0{let mut endpoints:Vec<_>=PHASES.iter().zip(r.ms.iter()).map(|(p,t)|serde_json::json!([p,t])).collect();endpoints.push(serde_json::json!(["next_route",r.ms[9]]));writeln!(f,"{}",serde_json::json!({"layer":r.layer,"rows":r.rows,"endpoints_ms":endpoints})).unwrap();}writeln!(f,"{}",serde_json::json!({"flush":true,"dropped":batch.1})).unwrap();f.flush().unwrap();}});buffer
 }) }
pub(crate) fn mark(stream:&Arc<CudaStream>,layer:usize,rows:usize,phase:&'static str){
 if !enabled(){return}let buffer=buffer();
 PENDING.with(|p|{let mut p=p.borrow_mut();
 if phase=="route" {let mut entry=FREE.with(|f|f.borrow_mut().pop()).unwrap_or_else(||Layer{layer,rows,count:0,events:(0..9).map(|_|stream.context().new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("partition event")).collect()});entry.layer=layer;entry.rows=rows;entry.count=0;p.push(entry);}
 let a=p.last_mut().expect("partition layer");assert_eq!(PHASES[a.count],phase);a.events[a.count].record(stream).expect("partition record");a.count+=1;
 // Two subsequent host routing reads ensure the first two route events completed.
 if phase=="route"&&p.len()>2{let a=&p[0];assert_eq!(a.count,9);let start=&a.events[0];let mut ms=[0.;10];for (i,e) in a.events.iter().enumerate(){ms[i]=start.elapsed_ms(e).expect("partition elapsed");}ms[9]=start.elapsed_ms(&p[1].events[0]).expect("partition span");let mut b=buffer.lock().unwrap();if b.rows.len()<65536{b.rows.push(Row{layer:a.layer,rows:a.rows,ms});}else{b.dropped+=1;}b.last=Instant::now();drop(b);let old=p.remove(0);FREE.with(|f|f.borrow_mut().push(old));}
 });
}
