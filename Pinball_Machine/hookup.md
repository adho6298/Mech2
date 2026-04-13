# Pinball Machine — Raspberry Pi 5 Wiring Reference

All GPIO numbers are **BCM** (Broadcom) numbering, as used by `gpiozero`.  
Physical (board) pin numbers are from the 40-pin header.  
Active-LOW inputs use the internal pull-up resistor.

---

## GPIO Pin Table

| BCM GPIO | Physical Pin | Direction | Active | Signal / Variable         | Connected To                                      |
|----------|-------------|-----------|--------|---------------------------|---------------------------------------------------|
| 2        | 3           | Output    | —      | I²C SDA                   | LCD1 SDA + LCD2 SDA (shared bus)                  |
| 3        | 5           | Output    | —      | I²C SCL                   | LCD1 SCL + LCD2 SCL (shared bus)                  |
| 5        | 29          | Input     | LOW    | `P1_DRAIN_PIN`            | P1 drain switch (ball exits P1 side)              |
| 6        | 31          | Input     | LOW    | `P2_DRAIN_PIN`            | P2 drain switch (ball exits P2 side)              |
| 9        | 21          | Input     | LOW    | `P1_LEFT_BTN_PIN`         | P1 (Red) Left flipper button                      |
| 10       | 19          | Output    | HIGH   | `P2_RIGHT_RELAY_PIN`      | P2 (Blue) Right flipper relay                     |
| 11       | 23          | Output    | HIGH   | `P1_LEFT_RELAY_PIN`       | P1 (Red) Left flipper relay                       |
| 15       | 10          | Output    | HIGH   | `P1_RIGHT_RELAY_PIN`      | P1 (Red) Right flipper relay                      |
| 16       | 36          | Input     | LOW    | `P1_RIGHT_BTN_PIN`        | P1 (Red) Right flipper button                     |
| 17       | 11          | Input     | LOW    | `P2_LEFT_BTN_PIN`         | P2 (Blue) Left flipper button (2P Human mode only)|
| 18       | 12          | Output    | HIGH   | `ARDUINO_P1_SCORE_PIN`    | Arduino digital input — P1 scored event           |
| 22       | 15          | Input     | LOW    | `P2_RIGHT_BTN_PIN`        | P2 (Blue) Right flipper button (2P Human mode only)|
| 23       | 16          | Output    | HIGH   | `ARDUINO_P2_SCORE_PIN`    | Arduino digital input — P2 scored event           |
| 24       | 18          | Output    | HIGH   | `ARDUINO_START_PIN`       | Arduino digital input — game start event          |
| 27       | 13          | Output    | HIGH   | `P2_LEFT_RELAY_PIN`       | P2 (Blue) Left flipper relay                      |


---

## I²C LCD Addresses

| Device | I²C Address | Notes                                    |
|--------|-------------|------------------------------------------|
| LCD1   | `0x27`      | Default PCF8574 address                  |
| LCD2   | `0x26`      | Solder the **A0** jumper on the backpack |

Both LCDs share the same SDA/SCL lines (pins 3 and 5).

---

## Arduino Signal Pins (RPi → Arduino)

The RPi pulses these pins HIGH for `0.1 s` on each game event.  
Wire each RPi GPIO output to a digital input pin on the Arduino.

| RPi GPIO | Physical Pin | Event Triggered  | Arduino reads…              |
|----------|-------------|------------------|-----------------------------|
| 18       | 12          | P1 scored        | Increment P1 LED/score count |
| 23       | 16          | P2 scored        | Increment P2 LED/score count |
| 24       | 18          | Game start       | Begin game-mode animation    |

---

## Power / Ground Reference

| Physical Pin | Function          |
|-------------|-------------------|
| 1 or 17     | 3.3 V             |
| 2 or 4      | 5 V               |
| 6, 9, 14, 20, 25, 30, 34, 39 | GND |

All relay modules and button pull-ups should share a common GND with the RPi.
