//! Experimental dispatcher. All worker acknowledgements precede release of the borrowed callback.
use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

fn aligned() -> bool {
    crate::config::runtime().cpu_physical_layout
}
fn pin(cpu: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        assert_eq!(libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set), 0, "pin CPU {cpu}: {}", std::io::Error::last_os_error());
    }
}
pub type Callback<'a> = &'a (dyn Fn(usize) + Sync + Send);
struct Borrowed<'a>(Callback<'a>);
struct State { epoch: usize, address: usize, stop: bool }
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    published: AtomicUsize,
    next: AtomicUsize,
    tasks: AtomicUsize,
    done: AtomicUsize,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}
struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    serial: Mutex<()>,
}
impl Pool {
    fn new(workers: usize, spin: Option<Duration>) -> Self {
        assert!(workers > 0);
        let shared = Arc::new(Shared {
            state: Mutex::new(State { epoch: 0, address: 0, stop: false }),
            wake: Condvar::new(), published: AtomicUsize::new(0),
            next: AtomicUsize::new(0), tasks: AtomicUsize::new(0), done: AtomicUsize::new(0),
            panic: Mutex::new(None),
        });
        let workers = (0..workers).map(|id| {
            let s = Arc::clone(&shared);
            thread::Builder::new().name(format!("moe-spin-{id}")).spawn(move || {
                if aligned() { pin(id + 1); }
                let mut seen = 0;
                loop {
                    let idle = Instant::now();
                    let mut polls = 0usize;
                    while s.published.load(Acquire) == seen {
                        polls = polls.wrapping_add(1);
                        if polls % 1024 == 0 && spin.map_or(false, |limit| idle.elapsed() >= limit) { break; }
                        std::hint::spin_loop();
                    }
                    let (epoch, address) = {
                        let mut st = s.state.lock().unwrap();
                        while !st.stop && st.epoch == seen { st = s.wake.wait(st).unwrap(); }
                        if st.stop { return; }
                        (st.epoch, st.address)
                    };
                    seen = epoch;
                    // The serialized caller keeps Borrowed alive until every worker acknowledges.
                    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
                        let callback = &*(address as *const Borrowed<'_>);
                        loop {
                            let i = s.next.fetch_add(1, Relaxed);
                            if i >= s.tasks.load(Relaxed) { break; }
                            (callback.0)(i);
                        }
                    }));
                    if let Err(panic) = result {
                        let mut failure = s.panic.lock().unwrap();
                        if failure.is_none() { *failure = Some(panic); }
                    }
                    s.done.fetch_add(1, Release);
                }
            }).expect("spawn CPU spin worker")
        }).collect();
        Self { shared, workers, serial: Mutex::new(()) }
    }
    fn run(&self, n: usize, f: Callback<'_>) {
        if n == 0 { return; }
        let guard = self.serial.lock().unwrap();
        if aligned() {
            thread_local! { static PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
            PINNED.with(|pinned| {
                if !pinned.replace(true) { pin(0); }
            });
        }
        let callback = Borrowed(f);
        let s = &self.shared;
        s.next.store(0, Relaxed);
        s.tasks.store(n, Relaxed);
        s.done.store(0, Relaxed);
        {
            let mut st = s.state.lock().unwrap();
            st.address = (&callback as *const Borrowed<'_>) as usize;
            st.epoch = st.epoch.checked_add(1).expect("dispatcher epoch overflow");
            s.published.store(st.epoch, Release);
            s.wake.notify_all();
        }
        if aligned() {
            let result = catch_unwind(AssertUnwindSafe(|| loop {
                let i = s.next.fetch_add(1, Relaxed);
                if i >= n { break; }
                f(i);
            }));
            if let Err(panic) = result { *s.panic.lock().unwrap() = Some(panic); }
        }
        while s.done.load(Acquire) != self.workers.len() { std::hint::spin_loop(); }
        let failure = s.panic.lock().unwrap().take();
        drop(guard);
        if let Some(panic) = failure { resume_unwind(panic); }
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.stop = true;
            st.epoch += 1;
            self.shared.published.store(st.epoch, Release);
            self.shared.wake.notify_all();
        }
        for worker in self.workers.drain(..) { worker.join().unwrap(); }
    }
}
pub fn dispatch(n: usize, f: Callback<'_>) {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cfg = crate::config::runtime();
        let workers = cfg.cpu_moe_threads;
        if aligned() { assert_eq!(workers, 5, "physical layout requires five workers plus caller"); }
        let spin = cfg.spin_budget();
        Pool::new(workers, spin)
    }).run(n, f);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "isolated dispatcher installation; run this test alone"]
    fn fused_quantized_output_matches_rayon_bits() {
        use candle_core::{quantized::{QTensor, GgmlDType}, Device, Tensor};
        use mistralrs_quant::gguf::fused_cpu::{fused_moe_decode, install_task_dispatcher};
        let mk = |n, k, seed: f32| {
            let v: Vec<_> = (0..4*n*k).map(|i| (i as f32*seed).sin()*0.1).collect();
            QTensor::quantize(&Tensor::from_vec(v, (4,n,k), &Device::Cpu).unwrap(), GgmlDType::Q4_0).unwrap()
        };
        let (g,u,d) = (mk(64,256,0.01), mk(64,256,0.02), mk(256,64,0.03));
        let x: Vec<_> = (0..3*256).map(|i| (i as f32*0.015).cos()).collect();
        let ids = [0,1,2,1,3,0,2,3,1];
        let weights = [0.5,0.3,0.2,0.4,0.6,0.0,0.2,0.3,0.5];
        let reference = fused_moe_decode(&g,&u,&d,&x,&ids,&weights,3).unwrap().unwrap();
        install_task_dispatcher(Some(dispatch));
        for _ in 0..100 {
            let actual = fused_moe_decode(&g,&u,&d,&x,&ids,&weights,3).unwrap().unwrap();
            assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
    }
    #[test]
    fn repeated_borrowed_batches_exactly_once() {
        for spin in [Some(Duration::ZERO), Some(Duration::from_micros(1000)), None] {
            let pool = Pool::new(4, spin);
            for batch in 0..200 {
                let values: Vec<_> = (0..batch % 37).map(|_| AtomicUsize::new(0)).collect();
                pool.run(values.len(), &|i| { values[i].fetch_add(1, Relaxed); });
                assert!(values.iter().all(|v| v.load(Relaxed) == 1));
                if batch % 50 == 0 { thread::sleep(Duration::from_millis(2)); }
            }
        }
    }
    #[test]
    fn panic_waits_for_workers_and_pool_remains_usable() {
        let pool = Pool::new(4, Some(Duration::from_micros(50)));
        assert!(catch_unwind(AssertUnwindSafe(|| pool.run(20, &|i| { if i == 3 { panic!("test") } }))).is_err());
        let done = AtomicUsize::new(0);
        pool.run(25, &|_| { done.fetch_add(1, Relaxed); });
        assert_eq!(done.load(Relaxed), 25);
    }
}
