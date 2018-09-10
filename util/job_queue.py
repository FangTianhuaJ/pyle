import threading
import Queue
import sys

debug=False
class JobResult(object):
    '''
    Hold a result from a function run in a thread.  A simple Future, but labrad Futures are tied into
    the twisted thread.
    '''
    def __init__(self, func, args, kw):
        self._func = func
        self._args = args
        self._kw = kw
        self.result = None
        self.exc_info = None
        self.complete = threading.Event()
    def get_result(self):
        self.complete.wait()
        if self.exc_info is not None:
            raise self.exc_info[0], self.exc_info[1], self.exc_info[2]
        else:
            return self.result

class JobQueue(object):
    '''
    A simple job queue for running tasks in a thread.  Submit by calling run_in_thread
    which returns a JobResult completion token.  Call get_result to wait on/retreive the
    result.  It only uses 1 thread.  It would be trivial to add multiple worker threads,
    but I doubt that gives any benefit in the intended use case: pre-fetching datasets from
    the datavault.
    '''
    def __init__(self):
        self.jobs = Queue.Queue()
        self.th = threading.Thread(target=self.thread_func)
        self.th.daemon = True
        self.th.start()

    def run_in_thread(self, func, *args, **kw):
        '''
        Submit a callable to the job queue.  
        '''
        jr = JobResult(func, args, kw)
        self.jobs.put(jr)
        return jr

    def thread_func(self):
        '''
        Worker thread main function.
        '''
        while True:
            j = self.jobs.get()
            try:
                j.result = j._func(*j._args, **j._kw)
            except BaseException as e:
                j.exc_info = sys.exc_info()
            del j._func # Delete these so they can be garbage collected.
            del j._args # We delete them before signaling completion so that
            del j._kw   # there is less chance of a user trying to use them (which would be a race)
            j.complete.set()
            del j # Also delete our local reference so the result can be gc'ed.
                  # Otherwise j will stick around until the next call to jobs.get() completes

