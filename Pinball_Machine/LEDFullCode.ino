#include <FastLED.h>

#define LED_PIN1     7
#define LED_PIN2     8
#define NUM_LEDS    20

CRGB led1[NUM_LEDS];
CRGB led2[NUM_LEDS];

uint8_t startColor = 0;
const uint8_t colorStep = 3; // Speed of color change
unsigned long lastUpdate = 0;
const unsigned long timeStep = 30; // Speed of movement (lower = faster)

int state = 0; 

void setup() {
  FastLED.addLeds<WS2812, LED_PIN1, GRB>(led1, NUM_LEDS);
  FastLED.addLeds<WS2812, LED_PIN2, GRB>(led2, NUM_LEDS);

}

void loop() {
  
  // STARTUP
  while (state == 0) {      // state 0 = startup
    for (int i = 0; i <= (NUM_LEDS - 1); i++) {
      led1[i] = CRGB ( 0, 0, 255);
      led2[i] = CRGB ( 0, 0, 255);
      FastLED.show();
      delay(60);
    }
    delay(50);
    for (int i = (NUM_LEDS - 1); i >= 0; i--) {
      led1[i] = CRGB ( 255, 0, 0);
      led2[i] = CRGB ( 255, 0, 0);
      FastLED.show();
      delay(60);
    }
  }
  
  // NORMAL PLAY
  while (state == 1) {  // state 1 = normal play
    unsigned long currentTime = millis();
    if (currentTime - lastUpdate >= timeStep) {
      lastUpdate = currentTime;

      // Shift all LEDs one step to the right to create the wave effect
      for (int i = NUM_LEDS - 1; i > 0; i--) {
        led1[i] = led1[i - 1];
        led2[i] = led2[i - 1];
      }
      
      // Insert new hue at the first LED
      led1[0] = CHSV(startColor, 255, 255);
      led2[0] = CHSV(startColor, 255, 255);
      startColor += colorStep; // Change color over time
      
      FastLED.show();
    }

  }

  // BLUE SCORE
  if (state == 2) { //state 2 = blue score 
    for (int j = 0; j <=3; j++) {
      for (int i = 0; i <= (NUM_LEDS - 1); i++) {
        led1[i] = CRGB ( 0, 0, 255);
        led2[i] = CRGB ( 0, 0, 255);
        FastLED.show();
        delay(20);
      }
      for (int i = 0; i <= (NUM_LEDS - 1); i++) {
        led1[i] = CRGB ( 0, 0, 0);
        led2[i] = CRGB ( 0, 0, 0);
        FastLED.show();
        delay(20);
      }
    }
    state = 1;
  }

  // RED SCORE
  if (state == 3) { //state 3 = red score 
    for (int j = 0; j <=3; j++) {
      for (int i = (NUM_LEDS - 1); i >= 0; i--) {
        led1[i] = CRGB ( 255, 0, 0);
        led2[i] = CRGB ( 255, 0, 0);
        FastLED.show();
        delay(20);
      }
      for (int i = (NUM_LEDS - 1); i >= 0; i--) {
        led1[i] = CRGB ( 0, 0, 0);
        led2[i] = CRGB ( 0, 0, 0);
        FastLED.show();
        delay(20);
      }
    }
    state = 1;
  }

}
