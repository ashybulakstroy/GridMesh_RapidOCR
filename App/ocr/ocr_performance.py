"""Per-job wall/process CPU timing, matching project 17's measurement units."""
from threading import Event, Thread
from time import perf_counter, process_time
import os
import psutil


class OCRPerformance:
    def __init__(self, publish=None):
        self.publish = publish or (lambda _: None)
        self.start_wall, self.start_cpu = perf_counter(), process_time()
        self.phase_wall, self.phase_cpu = self.start_wall, self.start_cpu
        self.name = None
        self.samples = []
        self.cpu_now = 0.0
        self.cpu_peak = 0.0
        self.system_cpu = 0.0
        self.system_peak = 0.0
        self.cpu_history = []
        self.stop_event = Event()
        self.thread = Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        proc = psutil.Process()
        proc.cpu_percent(None)
        psutil.cpu_percent(None)
        while not self.stop_event.wait(.2):
            self.cpu_now = proc.cpu_percent(None)
            self.cpu_peak = max(self.cpu_peak, self.cpu_now)
            self.system_cpu = psutil.cpu_percent(None)
            self.system_peak = max(self.system_peak, self.system_cpu)
            self.cpu_history.append({'elapsed_s':round(perf_counter()-self.start_wall, 2),
                                     'system_cpu_percent':self.system_cpu,
                                     'process_cpu_percent':self.cpu_now})
            self._publish()

    def _publish(self):
        self.publish({'phase': self.name, 'elapsed_s': round(perf_counter()-self.start_wall, 2),
                      'cpu_percent': self.cpu_now, 'cpu_peak_percent': self.cpu_peak,
                      'system_cpu_percent':self.system_cpu, 'system_cpu_peak_percent':self.system_peak,
                      'samples': list(self.samples), 'cpu_history': list(self.cpu_history)})

    def phase(self, name):
        now, cpu = perf_counter(), process_time()
        if self.name is not None:
            wall, used = now-self.phase_wall, cpu-self.phase_cpu
            self.samples.append({'name':self.name, 'wall_ms':round(wall*1000,2),
                                 'process_cpu_ms':round(used*1000,2),
                                 'cpu_core_equivalents':round(used/wall,2) if wall else 0})
        self.name, self.phase_wall, self.phase_cpu = name, now, cpu
        self._publish()

    def finish(self):
        self.phase(None)
        self.stop_event.set()
        self.thread.join(timeout=1)
        wall, used = perf_counter()-self.start_wall, process_time()-self.start_cpu
        return {'version':'ocr-performance-v1', 'wall_ms':round(wall*1000,2),
                'process_cpu_ms':round(used*1000,2),
                'cpu_core_equivalents':round(used/wall,2) if wall else 0,
                'peak_process_cpu_percent':self.cpu_peak, 'cpu_count':os.cpu_count(),
                'peak_system_cpu_percent':self.system_peak,
                'cpu_history':list(self.cpu_history),
                'samples':self.samples,
                'note':'CPU% процесса: 100% соответствует одному логическому ядру; '
                       'замер процесса включает все его потоки и может включать фоновую работу сервера.'}
