from precise_runner import PreciseEngine, PreciseRunner
from time import sleep
from subprocess import call

print('Initialize Engine')
#call(["adk-message-send", "led_start_pattern{pattern:13}"])
def on_act():
    print('wake word detected')
    #call(["adk-message-send", "led_start_pattern{pattern:13}"])
    #call(["posix-client", "kws_active"])

def on_pred(x):
    x = 1

# initiate precise engine with mycroft model
engine = PreciseEngine('precise-engine', 'indira.tflite')

# initiate precise runner that will listen, predict, and detect wakeword
runner = PreciseRunner(engine,  on_prediction=on_pred, on_activation=on_act)

# start runner
runner.start()
#sleep(35)
#call(["adk-message-send", "led_start_pattern{pattern:7}"])

# keep main thread active until user interrupt
try:
    while 1:
        print('listening...')
        sleep(60)
except:
    runner.stop()
