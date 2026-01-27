import RPi.GPIO as GPIO
import time

# Set up GPIO mode
GPIO.setmode(GPIO.BCM)

# Define the GPIO pin (GPIO 17, physical pin 11)
SERVO_PIN = 17

# Set up the servo pin as output
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Create PWM instance with 50Hz frequency (standard for servos)
pwm = GPIO.PWM(SERVO_PIN, 50)

# Start PWM with 0% duty cycle
pwm.start(0)

def set_angle(angle):
    """
    Set servo to a specific angle (0-180 degrees)
    Duty cycle calculation: 2% = 0°, 12% = 180°
    """
    duty = 2 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

try:
    print("Starting servo sweep...")
    print("Press Ctrl+C to stop")
    
    while True:
        # Sweep from 0 to 180 degrees
        print("Moving to 0°")
        set_angle(0)
        time.sleep(1)
        
        print("Moving to 45°")
        set_angle(45)
        time.sleep(1)
        
        print("Moving to 90°")
        set_angle(90)
        time.sleep(1)
        
        print("Moving to 135°")
        set_angle(135)
        time.sleep(1)
        
        print("Moving to 180°")
        set_angle(180)
        time.sleep(1)
        
        # Sweep back from 180 to 0 degrees
        print("Moving to 135°")
        set_angle(135)
        time.sleep(1)
        
        print("Moving to 90°")
        set_angle(90)
        time.sleep(1)
        
        print("Moving to 45°")
        set_angle(45)
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping servo...")

finally:
    # Clean up
    pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up. Program ended.")


