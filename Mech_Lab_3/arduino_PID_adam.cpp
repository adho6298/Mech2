#include <Servo.h>


/*SERVO POSITION CONSTANTS*/
/* These are used to set the position of the servo using "microseconds of the pulse width"
instead of degrees for finer control*/
#define SERVO_OFFSET 1425   //center position
#define SERVO_MIN 1000
#define SERVO_MAX 1900
  
/*REFERENCE RANGE*/
#define REFERENCE_MIN 0
#define REFERENCE_MAX 640
  
/*DELTA T*/
#define DT 0.0055  //seconds
  
/*PID PARAMETERS*/
#define Kp 0.7    //proportional coefficient
#define Ki 0.2    //integral coefficient
#define Kd 0.2    //derivative coefficient
  
/*UPSCALING TO Servo.writeMilliseconds*/
#define OUTPUT_UPSCALE_FACTOR 10
  
/*EMA ALPHAS*/
#define SENSOR_EMA_a 0.05
#define SETPOINT_EMA_a 0.01
  
/*SENSOR SPIKE NOISE HANDLING*/
/*Adjust these values to filter camera noise better*/
#define SENSOR_NOISE_SPIKE_THRESHOLD 100
#define SENSOR_NOISE_LP_THRESHOLD 300
  
float mapfloat(float x, float in_min, float in_max, float out_min, float out_max);
  
Servo myservo;
  
/*ARDUINO PINS*/
int pot_pin = 1;
int servo_pin = 3;
  
/*EMA VARIABLE INITIALIZATIONS*/
float sensor_filtered = 320;
int pot_filtered = 0;
  
/*GLOBAL SENSOR SPIKE NOISE HANDLING VARIABLES*/
int last_sensor_value = 320;  // Initialize to center to prevent rejecting first real data
int old_sensor_value = 320;
  
/*GLOBAL PID VARIABLES*/
float previous_error = 0;
float integral = 0;

/*DEBUG PRINTING*/
int debug_counter = 0;
#define DEBUG_PRINT_INTERVAL 20  // Print every 20 loops (~110ms at 5.5ms/loop)
  
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);  // Reduce timeout from default 1000ms to 1ms for non-blocking reads
  myservo.attach(servo_pin);
}
  
void loop() {
  /*START DELTA T TIMING*/
  unsigned long my_time = millis();
  
  /*READ POT AND RUN POT EMA*/
  int pot_value = analogRead(pot_pin);
  pot_filtered = (SETPOINT_EMA_a*pot_value) + ((1-SETPOINT_EMA_a)*pot_filtered);
    
  /*MAP POT POSITION TO CM SETPOINT RANGE*/
  float setpoint = mapfloat((float)pot_filtered, 0.0, 1024.0, REFERENCE_MIN, REFERENCE_MAX);
  float setpoint = 320;
  
  /*READ SENSOR DATA*/
  int sensor_value = last_sensor_value;  //default to last sensor value if no new data is available
  if (Serial.available() > 0){
    int temp = Serial.parseInt(); // Much faster and non-blocking with setTimeout(1)
    if(temp >= 0 && temp <= 640) { // Validate range before using
      sensor_value = temp;
    }
    // Clear any remaining bytes in buffer (including newline)
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
  
  /*REMOVE SENSOR NOISE SPIKES*/
  if(abs(sensor_value-old_sensor_value) < SENSOR_NOISE_LP_THRESHOLD && abs(sensor_value-last_sensor_value) < SENSOR_NOISE_SPIKE_THRESHOLD){  //everything is in order
    old_sensor_value = last_sensor_value;
    last_sensor_value = sensor_value;
  }
  else{                               //spike detected - set sample equal to last
    sensor_value = last_sensor_value;
  
  }
    
   /*RUN SENSOR EMA*/
  sensor_filtered = (SENSOR_EMA_a*sensor_value) + ((1-SENSOR_EMA_a)*sensor_filtered);
  
  /*PID CONTROLLER*/
  float error = setpoint - sensor_filtered;

  integral = integral + error*DT;

  float derivative = (error - previous_error)/DT;
  float output = (Kp*error + Ki*integral + Kd*derivative)*OUTPUT_UPSCALE_FACTOR;
  previous_error = error;
  
  // /*PRINT TO SERIAL THE TERM CONTRIBUTIONS*/
  // Serial.print("P: ");
  // Serial.print(Kp*error);
  // Serial.print('\t');
  // Serial.print("I: ");
  // Serial.print(Ki*integral);
  // Serial.print('\t');
  // Serial.print("D: ");
  // Serial.print(Kd*derivative);
  // Serial.print('\t');
  
  // /*PRINT TO SERIAL FILTERED VS UNFILTERED SENSOR DATA*/
  // Serial.print("Filtered sensor: ");
  // Serial.print(sensor_filtered);
  // Serial.print('\t');
  // Serial.print("Unfiltered sensor: ");
  // Serial.println(sensor_value);
  // Serial.print('\t');
  
  /*PREPARE AND WRITE SERVO OUTPUT*/
  int servo_output = round(output) + SERVO_OFFSET;
    
  if(servo_output < SERVO_MIN){ //saturate servo output at min/max range 
    servo_output = SERVO_MIN; 
  } 
  else if(servo_output > SERVO_MAX){
    servo_output = SERVO_MAX;
  }
    
  myservo.writeMicroseconds(servo_output);  //write to servo
  
  /*RATE-LIMITED DEBUG OUTPUT*/
  debug_counter++;
  if(debug_counter >= DEBUG_PRINT_INTERVAL){
    debug_counter = 0;
    
    // Convert servo position to degrees for easier reading
    float servo_degrees = mapfloat((float)servo_output, SERVO_MIN, SERVO_MAX, 0.0, 180.0);
    
    // Print key variables in clean format
    Serial.print("Ball: ");
    Serial.print(sensor_filtered, 1);
    Serial.print(" | Servo: ");
    Serial.print(servo_output, 1);
    Serial.print(" | Error: ");
    Serial.print(error, 1);
    Serial.print(" | P: ");
    Serial.print(Kp*error);
    Serial.print(" | I: ");
    Serial.print(Ki*integral);
    Serial.print(" | D: ");
    Serial.print(Kd*derivative);
    Serial.print(" | Loop(ms): ");
    Serial.println(millis() - my_time);
  }
  
  /*WAIT FOR DELTA T*/
  while(millis() - my_time < DT*1000);
}
  
float mapfloat(float x, float in_min, float in_max, float out_min, float out_max)
{
 return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}