//! Tail-only diagnostic: GPU event intervals, no stream synchronization.
use std::{cell::RefCell,sync::{OnceLock,mpsc},time::Instant};
use candle_core::{Tensor,cuda_backend::cudarc::driver::{CudaEvent,sys}};
struct Tail {events:Vec<CudaEvent>,count:usize,rows:usize,next_rows:usize,host:Instant}
#[derive(Default)]
struct Sample {begin:Option<Instant>,sum_ns:u64,calls:u64,device_greedy:u64,blocks:[u64;6],pipeline:std::collections::BTreeMap<&'static str,u64>,cpu_begin:Option<Instant>,cpu_ns:u64}
thread_local!{static SAMPLE:RefCell<Sample>=RefCell::new(Sample::default());}
pub(crate) fn host_mark(phase:&'static str){if !enabled(){return} SAMPLE.with(|s|{let mut s=s.borrow_mut();match phase {
 "pipeline_cpu_begin"=>s.cpu_begin=Some(Instant::now()),
 "pipeline_cpu_end"=>{if let Some(t)=s.cpu_begin.take(){s.cpu_ns+=t.elapsed().as_nanos() as u64;}},
 p if p.starts_with("pipeline_")=>*s.pipeline.entry(p).or_default()+=1,
 "sample_begin"=>{assert!(s.begin.is_none());s.begin=Some(Instant::now());},
 "sample_device_greedy"=>s.device_greedy+=1,
 "sample_end"=>{let start=s.begin.take().expect("sample begin");s.sum_ns+=start.elapsed().as_nanos() as u64;s.calls+=1;},"sample_block_cpu_input"=>s.blocks[5]+=1,"sample_block_logprobs"=>s.blocks[0]+=1,"sample_block_speculative"=>s.blocks[1]+=1,"sample_block_multi"=>s.blocks[2]+=1,"sample_block_temperature"=>s.blocks[3]+=1,"sample_block_processors"=>s.blocks[4]+=1,_=>unreachable!()}});}
thread_local!{static PENDING:RefCell<Vec<Tail>>=const{RefCell::new(Vec::new())};static FREE:RefCell<Vec<Tail>>=const{RefCell::new(Vec::new())};}
static TX:OnceLock<mpsc::SyncSender<serde_json::Value>>=OnceLock::new();
pub(crate) fn enabled()->bool {static ON:OnceLock<bool>=OnceLock::new();*ON.get_or_init(||std::env::var_os("NAIWA_HEAD_PARTITION_FILE").is_some())}
pub(crate) fn mark(xs:&Tensor,phase:&str){
 if !enabled()||!xs.device().is_cuda(){return}
 let tx=TX.get_or_init(||{let (tx,rx)=mpsc::sync_channel::<serde_json::Value>(4096);let file=std::fs::File::create(std::env::var_os("NAIWA_HEAD_PARTITION_FILE").unwrap()).expect("head file");std::thread::spawn(move||{use std::io::Write;let mut f=std::io::BufWriter::new(file);while let Ok(v)=rx.recv(){writeln!(f,"{v}").unwrap();f.flush().unwrap();}});tx});
 let stream=xs.device().as_cuda_device().unwrap().cuda_stream();
 PENDING.with(|p|{let mut p=p.borrow_mut();
 if phase=="tail_start"{let mut a=FREE.with(|f|f.borrow_mut().pop()).unwrap_or_else(||Tail{events:(0..4).map(|_|stream.context().new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap()).collect(),count:0,rows:0,next_rows:0,host:Instant::now()});a.count=0;a.rows=xs.elem_count()/xs.dims().last().copied().unwrap_or(1);a.host=Instant::now();p.push(a);}
 if phase=="next_model" && p.last().is_none_or(|a|a.count!=3){return;}
 let a=p.last_mut().unwrap();if phase=="next_model"{a.next_rows=xs.elem_count()/xs.dims().last().copied().unwrap_or(1);}
 let expected=["tail_start","head_start","head_end","next_model"];assert_eq!(expected[a.count],phase);a.events[a.count].record(&stream).unwrap();a.count+=1;
 if phase=="next_model"{SAMPLE.with(|s|{let mut s=s.borrow_mut();tx.try_send(serde_json::json!({"kind":"sampler_host","calls":s.calls,"device_greedy":s.device_greedy,"blocks":s.blocks,"pipeline":s.pipeline,"pipeline_cpu_ms":s.cpu_ns as f64/1e6,"total_ms":s.sum_ns as f64/1e6,"next_rows":a.next_rows})).expect("sample trace overflow");s.calls=0;s.device_greedy=0;s.sum_ns=0;s.blocks=[0;6];s.pipeline.clear();s.cpu_ns=0;});}
 if phase=="tail_start"&&p.len()>2{let a=&p[0];assert_eq!(a.count,4);let ms:Vec<_>=a.events.iter().map(|e|a.events[0].elapsed_ms(e).unwrap()).collect();tx.try_send(serde_json::json!({"rows":a.rows,"next_rows":a.next_rows,"endpoints_ms":ms,"host_tail_start_to_next_tail_ms":p[1].host.duration_since(a.host).as_secs_f64()*1000.})).expect("head trace overflow");let a=p.remove(0);FREE.with(|f|f.borrow_mut().push(a));}
 });
}
