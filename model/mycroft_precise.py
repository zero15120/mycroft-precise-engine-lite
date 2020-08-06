from precise_runner import PreciseEngine, PreciseRunner
from time import sleep

print('Initializing...')
def on_act():
    print('wake word detected')

def on_pred(x):
    x = 1

# initiate precise engine with mycroft model
engine = PreciseEngine('precise-engine', 'hey-mycroft.tflite')

# initiate precise runner that will listen, predict, and detect wakeword
runner = PreciseRunner(engine,  on_prediction=on_pred, on_activation=on_act)

# start runner
runner.start()

# keep main thread active until user interrupt
try:
    while 1:
        print('listening...')
        sleep(60)
except:
    runner.stop()
