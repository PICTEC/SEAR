"""
Tasks should be created:
- prepare datasets with noise
- for each model, run it on this dataset
- for each model, evaluate it
- aggregate the results

What is the scope? Should be defined...
"""

from abc import ABCMeta, abstractmethod

import atexit
import argparse
import dill
import os
import random
import socket
import subprocess
import sys
import threading
import time
import types
import unittest
import warnings


class Task(metaclass=ABCMeta):
    def __init__(self):
        self._successor_task = []
        self._prerequisites = []
        self._resource_locks = []
    
    @staticmethod
    def from_callable(cb):
        class AnonymousTask(Task):
            run = cb
        return AnonymousTask
    
    @abstractmethod
    def run(self):
        pass


class Deferred:
    """
    Should be passed
    """
    


class Worker:   # based on Q
    def __init__(self, name):
        self.name = str(name).encode("ascii")
        self.active = True
    
    def get_things(self):
        pass    
    
    def poll(self):
        """
        Return information whether the worker is free, waiting or working
        """
        
    def terminate(self):
        pass

        
class WorkerCoordinator:
    """
    There should be one coordinator on each machine, which brokers the resources.
    Created with first worker and torn down when there are no workers.
    
    TODO: probe whether the socket exists and if not exists, delete this object
    """
    
    _default_unix_socket = "/tmp/Q_class_coordinator.sock"

    def __init__(self, address):
        self.workers = []
        self.address = address
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(address)
        self.open = True
        atexit.register(self.close)
        self.thread = threading.Thread(target=self._cb)
        self.thread.daemon = False
        self.active = True
        self.thread.start()
        
    def close(self):
        self.sock.close()
        os.remove(self.address)
    
    def _cb(self):
        while self.active:
            self.sock.listen(3)
            conn, addr = self.sock.accept()
            thread = threading.Thread(target=self._handle_connection, args=(conn, addr))
            thread.daemon = True
            thread.start()
            time.sleep(0)
            
    def _handle_connection(self, conn, addr):
        state = None
        try:
            datagram = conn.recv(4096)
            while datagram:
                response, state = self._process_datagram(datagram, state)
                if response is not None:
                    conn.send(response)
                datagram = conn.recv(4096)
        finally:
            conn.close()
    
    def _process_datagram(self, datagram, state):
        if datagram == b"PROBE;":
            return b"OK;", state
        if state == None:
            if datagram.startswith(b"NEW_WORKER"):
                head, rest = datagram.split(b":", 1)
                if rest[:-1] == b"DEFAULT":
                    worker_id = self.create_worker()
                    return b"OK:" + worker_id + b";", None
                else:
                    raise NotImplementedError("Non-default workers not available")
            elif datagram.startswith(b"TERMINATE;"):
                self.active = False
                return None, None
            elif datagram.startswith(b"TERMINATE:"):
                head, rest = datagram.split(b":", 1)
                worker = [worker for worker in self.workers if worker.name == rest[:-1]]
                if worker:
                    worker[0].terminate()
                    return b"OK;", None
                else:
                    return b"NOK;", None
            else:
                raise ValueError("Incorrect command")
         
    
    @staticmethod
    def create_coordinator():
        coord = WorkerCoordinator(WorkerCoordinator._default_unix_socket)
        return coord
    
    def create_worker(self, id=None):
        w = Worker(random.randint(1, 1000000)) # TODO: should be GUID
        self.workers.append(w)
        return w.name


class WorkerCoordinatorReference:
    
    _default_unix_socket = "/tmp/Q_class_coordinator.sock"
    
    def __init__(self, address, auto_terminable=False):
        self.auto_terminable = auto_terminable
        self.address = address
        st = time.time()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        while (time.time() - st) < 5 or not os.path.exists(address):
            time.sleep(0.1)
        self.sock.connect(address)
    
    def __del__(self):
        if self.auto_terminable:
            self.sock.send(b"TERMINATE;")
        self.sock.close()
        self.sock.unlink()
    
    def try_new_worker(self, worker_object=None):
        if worker_object is None:
            self.sock.send(b"NEW_WORKER:DEFAULT;")
            msg = self.sock.recv(4096)
            assert msg.startswith(b"OK:")
            name = msg[3:-1]
            return WorkerReference(name=name, daemonic=self.auto_terminable, coordinator=self)
        else:
            raise NotImplementedError
    
    @classmethod
    def local_daemon(cls):
        if not cls.probe(cls._default_unix_socket):
            proc = subprocess.Popen([sys.executable, "run_experiments.py", "--new_coordinator"])
            return WorkerCoordinatorReference(cls._default_unix_socket, auto_terminable=True)
        else:
            return WorkerCoordinatorReference(cls._default_unix_socket, auto_terminable=False)
        
    @classmethod
    def probe(cls, address, port=None):
        if port is None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(address)
                sock.send(b"PROBE;")
                data = sock.recv(4096)
                assert data == b"OK;"
                return True
            except:
                return False
            finally:
                sock.close()
        else:
            raise NotImplementedError("No non-unix socket probing now")
    
    def terminate(self, name):
        self.sock.send(b"TERMINATE:" + bytes(name) + b";")
        data = self.sock.recv(4096)
        assert data.startswith(b"OK")
    
    
class WorkerReference:
    
    _managers = []
    
    def __init__(self, name=None, daemonic=False, coordinator=None):
        self.name = name
        self.daemonic = daemonic
        self.coordinator = coordinator
    
    @classmethod
    def default(cls):
        if not cls._managers:
            cls._managers.append(WorkerCoordinatorReference.local_daemon())
        for manager in cls._managers:
            ref = manager.try_new_worker()
            if ref:
                break
        else:
            raise RuntimeError("Cannot create a worker with currently known managers")
        return ref
    
    def terminate(self):
        self.coordinator.terminate(self.name)


class ExecutionEngine:
    def __init__(self):
        self.workers = []
        self.password = []

    def __del__(self):
        self.close()
        
    def create_local_worker(self, daemon=True):
        """
        If local worker is daemonic, it will terminate when ExecutionEngine is torn down
        """
        new_worker = WorkerReference.default()
        self.workers.append(new_worker)

    def schedule(self, task):
        serialized = dill.dumps(task)
        # push on the queue and wait for the reader to consume it

    def close(self):
        for worker_ref in self.workers:
            if worker_ref.daemonic:
                worker_ref.terminate()
    

class TestSingleWorkerTaskSubmission(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        if os.path.exists(WorkerCoordinatorReference._default_unix_socket):
            warnings.warn("Deleting existing default socket - the local daemon may be killed!")
            try:
                os.unlink(WorkerCoordinatorReference._default_unix_socket)
            except OSError:
                if os.path.exists(WorkerCoordinatorReference._default_unix_socket):
                    raise
        self.executor = ExecutionEngine()
        self.executor.create_local_worker(daemon=True)
        
    def test_existence_of_manager(self):
        manager = self.executor.managers[0]
        self.assertTrue(WorkerCoordinatorReference.probe(manager.address))

    def test_existence_of_worker(self):
        worker = self.executor.workers[0]
        self.assertTrue(worker.probe())
        ExecutorEngine.probe(self.executor.address)
        
    def test_submission_of_a_task(self):
        task = Task.from_callable(lambda self: print("xD"))
        self.executor.submit(task)

    def test_completion_of_a_task(self):
        task = Task.from_callable(lambda self: print("xD"))
        self.executor.submit(task)
        task.complete  # need to assert that
    
    def test_receiving_results(self):
        pass
    
    def test_failing_of_a_task(self):
        pass
        
    def test_pickling_dependencies(self):
        pass
    
    def test_passing_stubs(self):
        pass
        
    @classmethod
    def tearDownClass(self):
        self.executor.close()


class TestPickleability(unittest.TestCase):
    def test_base(self):
        task = Task.from_callable(lambda self: print("xD"))
        dill.check(task)
        
    def test_with_standard_module(self):
        task = Task.from_callable(lambda self: print(os.listdir(os.path.expanduser("~"))))
        dill.check(task)
        
    def test_with_nonstandard_module(self):
        import numpy as np
        task = Task.from_callable(lambda self: np.zeros([2, 3]))
        dill.check(task)
        
        
def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_coordinator", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = _get_args()
    if args.test:
        sys.argv = sys.argv[:1]
        unittest.main()
    if args.new_coordinator:
        WorkerCoordinator.create_coordinator()