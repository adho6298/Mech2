#include <Servo.h>

Servo myServo;
const int SERVO_PIN = 9;  // PWM pin for servo

void setup() {
  Serial.begin(1000000);  // Initialize serial communication
  myServo.attach(SERVO_PIN);  // Attach servo to pin 9
  myServo.write(90);  // Start at center position
  
  Serial.println("Arduino Servo Controller Ready");
  Serial.println("Send angle (0-180) to move servo");
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming angle as an integer
    int angle = Serial.parseInt();
    
    // Validate angle range
    if (angle >= 0 && angle <= 180) {
      myServo.write(angle);
      
      // Send confirmation back to Pi
      Serial.print("Moved to: ");
      Serial.println(angle);
    } else if (angle != 0 || Serial.peek() == '0') {
      // Only print error if we actually received invalid data
      Serial.println("Error: Angle must be 0-180");
    }
    
    // Clear any remaining newline characters
    while (Serial.available() > 0 && Serial.peek() == '\n' || Serial.peek() == '\r') {
      Serial.read();
    }
  }
}
