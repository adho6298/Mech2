#include <FastLED.h>

#define LED_PIN1     9
#define LED_PIN2     8
#define NUM_LEDS    32

CRGB led1[NUM_LEDS];
CRGB led2[NUM_LEDS];

uint8_t startColor = 0;
const uint8_t colorStep = 3; // Speed of color change
unsigned long lastUpdate = 0;
const unsigned long timeStep = 30; // Speed of movement (lower = faster)

const uint16_t pulsePeriodMs = 2000; // full pulse cycle length (ms)
const uint8_t minBrightness = 50;    // lowest brightness (0-255)
const uint8_t maxBrightness = 255;   // highest brightness (0-255)
const CRGB blueColor = CRGB::Blue;
const CRGB redColor = CRGB::Red;
const unsigned long winduration = 7000;
unsigned long wintime;

int state = 0; 
const int lwire_pin = 2;
const int redpin =5;
const int bluepin =6;
const int startpin =4;
const int startout = 11;
const int winout = 12;
const int scoreout = 13;
int count1 = 0;
const int wincount=3;
int redcount =0;
int bluecount =0;

void setup() {
  FastLED.addLeds<WS2812, LED_PIN1, GRB>(led1, NUM_LEDS);
  FastLED.addLeds<WS2812, LED_PIN2, GRB>(led2, NUM_LEDS);

  pinMode(lwire_pin, OUTPUT);
  pinMode(redpin, INPUT);
  pinMode(bluepin, INPUT);
  pinMode(startpin, INPUT);
  pinMode(startout, OUTPUT);
  pinMode(winout, OUTPUT);
  pinMode(scoreout, OUTPUT);

  for (int i = 0; i < 3; i++){
    digitalWrite(lwire_pin, HIGH);
    delay(50);
    digitalWrite(lwire_pin, LOW);
    delay(1000);
    digitalWrite(lwire_pin, HIGH);
    delay(100);
    digitalWrite(lwire_pin, LOW);
    delay(150);
    digitalWrite(lwire_pin, HIGH);
    delay(50);
    digitalWrite(lwire_pin, LOW);
    delay(500);
    digitalWrite(lwire_pin, HIGH);
    delay(100);
    digitalWrite(lwire_pin, LOW);
    delay(100);
  }
  digitalWrite(lwire_pin, HIGH);

}

void loop() {
  
  // STARTUP
  while (state == 0) {      // state 0 = startup

    for (int i = 0; i <= (NUM_LEDS - 1); i++) {
      led1[i] = CRGB ( 0, 0, 230);
      led2[i] = CRGB ( 0, 0, 230);
      FastLED.show();

      int startgame = digitalRead(startpin);
      if (startgame == HIGH) { //start game
        state = 1;
        break;
      }
      delay(30);
    }
    if (state ==1){
      break;
    }
    for (int i = (NUM_LEDS - 1); i >= 0; i--) {
      led1[i] = CRGB ( 230, 0, 0);
      led2[i] = CRGB ( 230, 0, 0);
      FastLED.show();

      int startgame = digitalRead(startpin);
      if (startgame == HIGH) { //start game
        state = 1;
        digitalWrite(startout, HIGH);
        break;
      }
      delay(30);
    }
    if (state ==1){
      break;
    }
    if (count1 == 5) {
      digitalWrite(lwire_pin, LOW);
      delay(50);
      digitalWrite(lwire_pin, HIGH);
      count1=0;
    }
    count1++;

    int startgame = digitalRead(startpin);
    if (startgame == HIGH) { //start game
      state = 1;
      digitalWrite(startout, HIGH);
    }
  }
  
  // NORMAL PLAY
  while (state == 1) {  // state 1 = normal play
    digitalWrite(lwire_pin, HIGH);
    unsigned long currentTime = millis();
    if (currentTime - lastUpdate >= timeStep) {
      lastUpdate = currentTime;

      // Shift all LEDs one step to the right to create the wave effect
      for (int i = NUM_LEDS - 1; i > 0; i--) {
        led1[i] = led1[i - 1];
      }
      for (int i = 0; i <= (NUM_LEDS - 1); i++) {
        led2[i] = led2[i + 1];
      }

      // Insert new hue at the first LED
      led1[0] = CHSV(startColor, 255, 255);
      led2[NUM_LEDS] = CHSV(startColor, 255, 255);
      startColor += colorStep; // Change color over time
      
      FastLED.show();
    }

    digitalWrite(startout, LOW);
    int blue = digitalRead(bluepin);
    int red = digitalRead(redpin);
    if (blue == HIGH) { //blue score signal
      state = 2;
    }
    if (red == HIGH) { //red score signal
      state = 3;
    }
  }

  // BLUE SCORE
  if (state == 2) { //state 2 = blue score
    bluecount=bluecount+1;
    if (bluecount < wincount){
      digitalWrite(scoreout, HIGH);
      fill_solid(led1, NUM_LEDS, CRGB::Black);
      fill_solid(led2, NUM_LEDS, CRGB::Black);
      for (int j = 0; j < 3; j++) {
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
      digitalWrite(lwire_pin, HIGH);
      digitalWrite(scoreout, LOW);
      state = 1;
    }
    //BLUE WIN
    else if (bluecount == wincount){
      digitalWrite(winout, HIGH);
      digitalWrite(lwire_pin, LOW);
      fill_solid(led1, NUM_LEDS, CRGB::Black);
      fill_solid(led2, NUM_LEDS, CRGB::Black);

      for (int j = 0; j < 3; j++) {
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

      wintime = millis();

      while (millis() - wintime < winduration) {
        // compute a smooth 0..1 value using sine
        uint32_t t = millis() % pulsePeriodMs;
        float phase = (float)t / pulsePeriodMs;               // 0..1
        float sine = 0.5f * (1.0f + sinf(phase * 2.0f * PI)); // 0..1 smooth
        // map to brightness range
        uint8_t bri = minBrightness + (uint8_t)((maxBrightness - minBrightness) * sine);
        // set color with brightness applied
        CRGB color = blueColor;
        color.nscale8_video(bri); // scale color by brightness
       
        fill_solid(led1, NUM_LEDS, color);
        fill_solid(led2, NUM_LEDS, color);
        FastLED.show();
      
        delay(10);
      }
      
      bluecount = 0;
      redcount = 0;
      state = 0;
      digitalWrite(winout, LOW);
      digitalWrite(lwire_pin, HIGH);
      fill_solid(led1, NUM_LEDS, CRGB::Black);
      fill_solid(led2, NUM_LEDS, CRGB::Black);
      delay(100);
    }
  }

  // RED SCORE
  if (state == 3) { //state 3 = red score
    redcount = redcount+1;
    if (redcount < wincount) {
      digitalWrite(scoreout, HIGH);
      fill_solid(led1, NUM_LEDS, CRGB::Black);
      fill_solid(led2, NUM_LEDS, CRGB::Black);
      for (int j = 0; j < 3; j++) {
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
      digitalWrite(scoreout, LOW);
      state = 1;
    }
    else if (redcount == wincount) {
      digitalWrite(winout, HIGH);
      digitalWrite(lwire_pin, LOW);
      fill_solid(led1, NUM_LEDS, CRGB::Black);
      fill_solid(led2, NUM_LEDS, CRGB::Black);

      for (int j = 0; j < 3; j++) {
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

      wintime = millis();

      while (millis() - wintime < winduration) {
        // compute a smooth 0..1 value using sine
        uint32_t t = millis() % pulsePeriodMs;
        float phase = (float)t / pulsePeriodMs;               // 0..1
        float sine = 0.5f * (1.0f + sinf(phase * 2.0f * PI)); // 0..1 smooth
        // map to brightness range
        uint8_t bri = minBrightness + (uint8_t)((maxBrightness - minBrightness) * sine);
        // set color with brightness applied
        CRGB color = redColor;
        color.nscale8_video(bri); // scale color by brightness
        
        fill_solid(led1, NUM_LEDS, color);
        fill_solid(led2, NUM_LEDS, color);
        FastLED.show();
        
        delay(10);
      }
      
      bluecount = 0;
      redcount = 0;
      state = 0;
      digitalWrite(winout, LOW);
      digitalWrite(lwire_pin, HIGH);
      fill_solid(led1, NUM_LEDS, CRGB::Black);
      fill_solid(led2, NUM_LEDS, CRGB::Black);
      delay(100);
    }
  }

}
