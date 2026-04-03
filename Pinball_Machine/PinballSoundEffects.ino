// VARIABLE SETUP
int state = 0;
const int spk = 11;
unsigned long millis1 = millis();


// SOUND FUNCTION LIBRARY
void toneMs(int freq, int ms) { //single tone
  if (freq <= 0) { delay(ms); return; }
  tone(spk, freq);
  delay(ms);
  noTone(spk);
}

void glide(int a, int b, int totalMs, int steps) { // Smooth frequency glide
  for (int i = 0; i <= steps; ++i) {
    int f = a + (b - a) * i / steps;
    tone(spk, f);
    delay(totalMs / steps);
  }
  noTone(spk);
}

void powerUpSequence() { //STARTUP NOISES FUNCTION
  // 1) Soft startup hum rising slightly (background)
  glide(60, 120, 4000, 60);

  // 2) Short silence
  delay(120);

  // 3) Rapid ascending boot beeps (like digital checks)
  int beepBase = 400;
  for (int i = 0; i < 5; ++i) {
    int f = beepBase + i * 160 + (i%2==0 ? 0 : 30); // slight alternation
    toneMs(f, 90 - i*8);
    delay(100);
  }

  // 4) Brief glitch/diagnostic stutter
  for (int g = 0; g < 3; ++g) {
    toneMs(1200 - g*120, 40);
    delay(30);
  }
  delay(100);

  // 5) Warm power-on chime
  glide(300, 1400, 220, 40);

  // 6) Two-tone hum
  delay(50);
  toneMs(220, 200);
  toneMs(500, 200);
  delay(50);

  // 7) Final confirmation blip
  toneMs(1000, 120);

  delay (50);
  noTone(spk);
}

void blip(int startHz, int endHz, int steps, int stepDelay) {
  for (int i = 0; i <= steps; ++i) {
    int f = startHz + (endHz - startHz) * i / steps;
    tone(spk, f);
    delay(stepDelay);
  }
  noTone(spk);
}

void laser(int startHz, int endHz, int steps, int stepDelay) {
  for (int i = 0; i <= steps; ++i) {
    int f = startHz + (endHz - startHz) * i / steps;
    tone(spk, f);
    delay(stepDelay);
  }
  // short click
  tone(spk, 1200, 40);
  delay(60);
  noTone(spk);
}

// void swoosh(int startHz, int endHz, int steps, int stepMs) {
//   for (int i = 1; i <= steps; i++) {
//     int f = startHz + ((endHz - startHz) * i / steps);
//     tone(spk, f);
//     delay(stepMs * 0.7);
//     noTone(spk);
//     delay(stepMs * 0.3);
//   }
// }

void alarm(int f1, int f2, int pulseMs, int cycles) {
  for (int j = 0; j < cycles; ++j) {
    tone(spk, f1);
    delay(pulseMs);
    tone(spk, f2);
    delay(pulseMs);
    // tremolo: rapid on/off to thicken
    for (int k = 0; k < 4; ++k) {
      noTone(spk); delay(10);
      tone(spk, f2); delay(10);
    }
  }
  noTone(spk);
}

void vibratoTone(int baseHz, int depthHz, int rateMs, int durationMs) {
  unsigned long end = millis() + durationMs;
  bool sign = false;
  while (millis() < end) {
    int f = baseHz + (sign ? depthHz : -depthHz);
    tone(spk, f);
    sign = !sign;
    delay(rateMs);
  }
  noTone(spk);
}

void raygun(int baseHz, int peakHz, int lengthMs) {
  unsigned long start = millis();
  unsigned long end = start + lengthMs;
  while (millis() < end) {
    float t = float(millis() - start) / lengthMs; // 0.0 -> 1.0
    // Frequency envelope: quick rise then slow fall (nonlinear)
    float env = (t < 0.25) ? (t / 0.25) : pow(1.0 - (t - 0.25) / 0.75, 1.5);
    float freq = baseHz + env * (peakHz - baseHz);
    // Add subtle vibrato for "wobble"
    float vib = 15.0 * sin(2.0 * PI * 6.0 * t); // 6 Hz vibrato, 15 Hz depth
    tone(spk, int(freq + vib));
    // Pulse-width envelope to simulate amplitude swell (pseudo-volume)
    int onMs = (int)(4 + 26 * env); // more on-time when env is larger
    delay(onMs);
    noTone(spk);
    delay(6); // short off-time to shape amplitude
  }
  noTone(spk);
}


void setup() {
  pinMode(7, INPUT_PULLUP);
  pinMode(8, INPUT_PULLUP);

  // STARTUP NOISES
  //powerUpSequence();
  delay(1000);
  tone(spk,200);
  delay(100);
  tone(spk,100);
  delay(75);
  noTone(spk);
}

void loop() {
  
  unsigned long startmillis = millis();

  // STANDBY MODE
  while (state == 0){ // small beebing every few seconds
    if (millis() - startmillis > 4000){
      tone(spk,200);
      delay(100);
      tone(spk,100);
      delay(75);
      noTone(spk);
      startmillis = millis();
    }
    else{
      noTone(spk);
    }

    // check state to get out of while loop
    int yellow = digitalRead(2);
    if (yellow == LOW) {
      state =1;
    }
  }


  // PLAY MODE
  if (state == 1){
    if (millis() - millis1 > 4000){ //play random sound effects every few seconds
      long rand = random(8);
      //int rand =0;

      if (rand == 1) {
        blip(2500, 800, 20, 8); // high quick blip down
      }

      else if (rand == 2) {
        laser(400, 5000, 40, 4); // beep up
      }

      else if (rand == 3) {
        glide(100, 500, 220, 40);
      }

      else if (rand == 4) {
        laser(200, 1000, 40, 4);
      }
      
      else if (rand == 5) {
        vibratoTone(400, 10, 50, 300); // subtle vibrato
      }
      
      else if (rand == 6) {
        vibratoTone(800, 10, 50, 300);
      }
      
      else if (rand == 7) {
        raygun(500,1400,200);
      }

      if (rand == 0) {
        glide(300, 1000, 220, 40);
      }

      millis1 = millis();
    }
    else{
      noTone(spk);
    }

  }

  // SCORE CONDITION
  if (state == 2){ //score
    alarm(600, 900, 200, 2);
    delay(50);
    for (int i=0; i<3; i++) {
      raygun(130, 1600, 400);
      delay(180);
    }
    //tone(spk,150,1000);
    delay (500);

  }

  // WIN CONDITION
  if (state == 3){ //win
    alarm(600, 900, 200, 4);
    vibratoTone(800, 50, 100, 3000);
    delay(100);
    tone(spk,1000);
    delay (200);
    noTone(spk);
    delay(75);
    tone(spk,1000);
    delay (200);
    noTone(spk);
    delay(75);
    tone(spk,1000);
    delay (200);
    noTone(spk);
    delay(75);
    tone(spk,1000);
    delay (200);
    noTone(spk);
    delay(75);
    tone(spk,1000);
    delay (200);
    noTone(spk);
    delay(1000);

    // return to standby
    state = 0;
  }

  // CHECK STATE
  int red = digitalRead(8);
  int green = digitalRead(7);

  if (red == LOW){
    state = 2;
  }
  else if (green == LOW){
    state = 3;
  }
  else if (red == HIGH && green == HIGH) {
    state =1;
  }
}