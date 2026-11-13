use rayon::prelude::*;
use std::{
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering::*},
        Arc, Condvar, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
struct State {
    epoch: usize,
    stop: bool,
}
struct Shared {
    published: AtomicUsize,
    state: Mutex<State>,
    cv: Condvar,
    next: AtomicUsize,
    done: AtomicUsize,
    tasks: usize,
    work: usize,
    out: Vec<AtomicU64>,
}
fn calc(i: usize, n: usize) -> u64 {
    let mut x = i as u64 + 1;
    for _ in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        x ^= x >> 17;
    }
    std::hint::black_box(x)
}
fn drain(s: &Shared) {
    loop {
        let i = s.next.fetch_add(1, Relaxed);
        if i >= s.tasks {
            break;
        }
        s.out[i].store(calc(i, s.work), Relaxed);
    }
}
struct Flat {
    s: Arc<Shared>,
    threads: Vec<thread::JoinHandle<()>>,
    host: bool,
}
impl Flat {
    fn new(workers: usize, host: bool, tasks: usize, work: usize, spin: u64) -> Self {
        let s = Arc::new(Shared {
            published: AtomicUsize::new(0),
            state: Mutex::new(State {
                epoch: 0,
                stop: false,
            }),
            cv: Condvar::new(),
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            tasks,
            work,
            out: (0..tasks).map(|_| AtomicU64::new(0)).collect(),
        });
        let threads = (0..workers)
            .map(|_| {
                let s = s.clone();
                thread::spawn(move || {
                    let mut seen = 0;
                    loop {
                        let start = Instant::now();
                        loop {
                            let epoch = s.published.load(Acquire);
                            if epoch != seen {
                                seen = epoch;
                                break;
                            }
                            if start.elapsed() < Duration::from_micros(spin) {
                                std::hint::spin_loop();
                            } else {
                                let mut st = s.state.lock().unwrap();
                                while !st.stop && st.epoch == seen {
                                    st = s.cv.wait(st).unwrap();
                                }
                                if st.stop {
                                    return;
                                }
                                seen = st.epoch;
                                break;
                            }
                        }
                        drain(&s);
                        s.done.fetch_add(1, Release);
                    }
                })
            })
            .collect();
        Self { s, threads, host }
    }
    fn run(&self) {
        self.s.next.store(0, Relaxed);
        self.s.done.store(0, Relaxed);
        {
            let mut st = self.s.state.lock().unwrap();
            st.epoch += 1;
            self.s.published.store(st.epoch, Release);
            self.s.cv.notify_all();
        }
        if self.host {
            drain(&self.s)
        }
        while self.s.done.load(Acquire) < self.threads.len() {
            std::hint::spin_loop();
        }
    }
}
impl Drop for Flat {
    fn drop(&mut self) {
        {
            let mut st = self.s.state.lock().unwrap();
            st.stop = true;
            self.s.cv.notify_all();
        }
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = &args[1];
    let n: usize = args[2].parse().unwrap();
    let tasks: usize = args[3].parse().unwrap();
    let work: usize = args[4].parse().unwrap();
    let gap: u64 = args[5].parse().unwrap();
    let pool = (mode == "rayon").then(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
    });
    let out: Vec<_> = (0..tasks).map(|_| AtomicU64::new(0)).collect();
    let flat = if mode == "rayon" {
        None
    } else {
        Some(Flat::new(
            n - usize::from(mode == "host"),
            mode == "host",
            tasks,
            work,
            50,
        ))
    };
    let mut times = Vec::new();
    for k in 0..350 {
        if gap > 0 {
            thread::sleep(Duration::from_micros(gap));
        }
        let t = Instant::now();
        if let Some(f) = &flat {
            f.run()
        } else {
            pool.as_ref().unwrap().install(|| {
                (0..tasks)
                    .into_par_iter()
                    .for_each(|i| out[i].store(calc(i, work), Relaxed))
            });
        }
        let us = t.elapsed().as_secs_f64() * 1e6;
        if k >= 50 {
            times.push(us);
        }
    }
    let result = flat.as_ref().map(|f| &f.s.out).unwrap_or(&out);
    for i in 0..tasks {
        assert_eq!(result[i].load(Relaxed), calc(i, work));
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{mode},{n},{tasks},{work},{gap},{:.3},{:.3},{:.3}",
        times[150], times[270], times[297]
    );
}
