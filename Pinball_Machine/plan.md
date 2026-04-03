# Pinball Machine — Full System Integration Plan

## Project Overview

A 2-player pinball machine built on a **Raspberry Pi 5**. Gameplay is competitive — like air hockey or foosball with pinball mechanics. Each player defends their side with flippers; when the ball gets past a player's flippers and drains on their side, the **other player scores a point**. Player 1 is always human-operated. Player 2 can be human-operated or controlled by a computer vision system that detects the ball and actuates the flippers automatically.

### Hardware Summary
| Component | Details |
|---|---|
| Brain | Raspberry Pi 5 |
| Displays | 2× I2C IIC LCD 1602 (0x27, 0x26) on shared SDA/SCL |
| P1 Buttons | 2× momentary push buttons (left/right flipper) |
| P2 Buttons | 2× momentary push buttons (left/right flipper, human mode only) |
| Drain Switches | 2× limit switches — one per side; ball draining on P1's side = P2 scores, and vice versa |
| Camera | Raspberry Pi Camera (PiCamera2) for CV flipper actuation |
| Flipper Relays | 4× relays (P1 left/right, P2 left/right) |
| Arduino (LED) | Controls EL wire + individually addressable LEDs (WS2812) |
| Arduino (Sound) | Controls sound effects |

---

## Existing Code

| File | Status | Description |
|---|---|---|
| [Score.py](Score.py) | Usable, refactor into `main.py` | LCD display, limit switch debounce, win condition, simulation fallback — drain switch logic replaces its score switch callbacks |
| [Pinball_Camera.py](Pinball_Camera.py) | Usable, wrap as thread | PiCamera2 + OpenCV brightness-threshold ball detection; triggers P2 flipper relays |
| [display_test.py](display_test.py) | Keep as standalone test | LCD boot test; helper init/write functions to reuse |
| [LEDFullCode.ino](LEDFullCode.ino) | Extend | FastLED Arduino; currently reads a `state` int variable |
| [requirements.txt](requirements.txt) | Update as needed | picamera2, opencv, RPLCD, gpiozero, lgpio, pynput |

### Lab Code for Reference
| File | Relevance |
|---|---|
| [../Mech_Lab_3/ball_detection_fast.py](../Mech_Lab_3/ball_detection_fast.py) | Threaded capture pattern; HSV-based detection alternative if brightness threshold is unreliable |
| [../Mech_Lab_3/PID_Camera.py](../Mech_Lab_3/PID_Camera.py) | Camera loop reference |

---

## GPIO Pin Map

| Function | GPIO (BCM) | Physical Pin | Notes |
|---|---|---|---|
| I2C SDA | 2 | 3 | Both LCDs shared |
| I2C SCL | 3 | 5 | Both LCDs shared |
| P1 drain switch (P2 scores) | 22 | 15 | Existing — repurposed from `Score.py` score switch |
| P2 drain switch (P1 scores) | 23 | 16 | Existing — repurposed from `Score.py` score switch |
| P2 Left flipper relay | 17 | 11 | Existing — `Pinball_Camera.py` |
| P2 Right flipper relay | 27 | 13 | Existing — `Pinball_Camera.py` |
| P1 Left button (input) | TBD-A | — | New |
| P1 Right button (input) | TBD-B | — | New |
| P2 Left button (input) | TBD-C | — | New — human P2 mode only |
| P2 Right button (input) | TBD-D | — | New — human P2 mode only |
| P1 Left flipper relay (output) | TBD-E | — | New — suggest GPIO 5 (pin 29) |
| P1 Right flipper relay (output) | TBD-F | — | New — suggest GPIO 6 (pin 31) |
| Arduino LED signal | TBD | — | Communication method TBD |
| Arduino Sound trigger | TBD | — | Integration deferred |

> **Note:** Confirm all TBD pins against your full RPi 5 pinout. Avoid GPIO 0/1 (reserved), 14/15 (UART), and any pins used by PiCamera2.

---

## Architecture

### File Structure (Target)
```
Pinball_Machine/
    main.py            ← Single entry point; game state machine; all subsystem coordination
    lcd.py             ← LCD abstraction layer (extracted from Score.py + display_test.py)
    LEDFullCode.ino    ← Extend with states 4–7 and binary GPIO input reading
    Pinball_Camera.py  ← Keep as reference; logic moved into CameraThread class in main.py
    Score.py           ← Keep as reference; logic moved into main.py
    display_test.py    ← Keep as standalone hardware test
    requirements.txt   ← Update as needed
    plan.md            ← This document
```

### Threading Model
- `main.py` runs the **game state machine** on the main thread
- `CameraThread` runs the **ball detection loop** on a background thread
- Thread starts on `GAME_START → GAMEPLAY` (CV mode only)
- Thread stops cleanly on `GAME_OVER` or program exit
- Ball detection callback fires `on_ball_detected(side: str)` → main thread checks game state before actuating relay

---

## Game State Machine

```
Boot
 └─→ ATTRACT
      └─→ MODE_SELECT  (any button press)
           └─→ GAME_START  (both P1 buttons held simultaneously)
                └─→ GAMEPLAY  (after 3s countdown)
                     ├─→ POINT_SCORED  (drain switch fires — opponent gets point, 2s pause)
                     │    └─→ GAMEPLAY  (score limit not yet reached)
                     └─→ GAME_OVER  (score limit reached)
                          └─→ ATTRACT  (after win display timeout)
```

### State Descriptions

| State | LCD Display | LED State | Camera Thread |
|---|---|---|---|
| `ATTRACT` | Attract animation / "PRESS START" | TBD | Off |
| `MODE_SELECT` | Mode options + selection highlight | TBD | Off |
| `GAME_START` | Countdown: 3… 2… 1… | TBD | Off |
| `GAMEPLAY` | P1 score left / P2 score right | TBD | On (CV mode only) |
| `POINT_SCORED` | "P2 SCORES! P1:X P2:Y" (2s pause) | TBD | Paused |
| `GAME_OVER` | "PLAYER X WINS!" flashing | TBD | Off |

---

## Arduino LED & Sound

Details TBD. Game events that will need LED/sound responses: point scored, game start, win, attract/idle. Communication method and pin assignments to be defined when Arduino sketches are finalized.

---

## Implementation Phases

### Phase 1 — Architecture & Config *(start here)*
- [ ] Create `main.py` with a `GameState` enum and a single config block at the top (all GPIO pins, timing constants, score limit)
- [ ] Confirm all TBD GPIO pins and fill in the config block
- [ ] Skeleton `run()` loop that transitions between states and prints state name to console

### Phase 2 — LCD Layer
- [ ] Create `lcd.py` — extract and consolidate from `Score.py` and `display_test.py`
- [ ] Implement: `init_lcds()`, `write_to_lcds(line1, line2)`, `show_attract()`, `show_mode_select(options, selected_idx)`, `show_countdown(n)`, `show_scores(p1, p2)`, `show_point_scored(scorer, p1, p2)`, `show_winner(player)`
- [ ] All functions must include simulation fallback (print to console when hardware libs unavailable)
- [ ] Both LCDs always display the same content simultaneously

### Phase 3 — Input Handling
- [ ] Centralize all GPIO inputs in `main.py` using `gpiozero.Button` + `LGPIOFactory` (pattern from `Score.py` `init_switches()`)
- [ ] **P1/P2 flipper buttons**: Hold behavior — relay `HIGH` while button held, `LOW` on release. Use `button.when_pressed` / `button.when_released`. Solenoids are rated for sustained hold; no max-hold timer needed.
- [ ] **Mode select**: P1 left/right buttons cycle between `"1P vs CPU"` and `"2P Human"` on the LCD; pressing both simultaneously confirms. CV mode is automatically enabled when `"1P vs CPU"` is selected and never runs in `"2P Human"` mode.
- [ ] **Drain switches**: Trigger `on_ball_drain(drained_player)` → increment score for the **opposing** player with debounce (from `Score.py` debounce pattern) → transition to `POINT_SCORED`

### Phase 4 — Arduino Signaling *(details TBD)*
- [ ] Stub Arduino signal pins in config block once communication method is decided
- [ ] Add placeholder calls at each game event (point scored, game start, win) for later integration

### Phase 5 — Camera Thread
- [ ] Wrap `Pinball_Camera.py` detection loop into a `CameraThread` class in `main.py`
- [ ] Expose `start()`, `stop()`, and `on_ball_detected(side: str)` callback
- [ ] Callback fires GPIO 17 (left) or GPIO 27 (right) only when `game_state == GAMEPLAY` and `p2_mode == "CV"` (i.e. 1P vs CPU mode was selected)
- [ ] Thread starts on transition into `GAMEPLAY` (CV mode only); stops on `GAME_OVER`

### Phase 6 — Scoring System
- [ ] Config: `SCORE_LIMIT = 10` (first player to reach this wins; `0` = no limit)
- [ ] Track `scores = {"P1": 0, "P2": 0}` in global game state
- [ ] On drain: increment opponent's score, enter `POINT_SCORED` state, show updated scores on LCD for 2s, then return to `GAMEPLAY`
- [ ] If either player reaches `SCORE_LIMIT`: transition to `GAME_OVER`; winner is the player who reached the limit

### Phase 7 — Full Integration & Cleanup
- [ ] Connect all phases into a single `main.py` `run()` loop
- [ ] Ensure all GPIO is cleaned up on `SIGINT` and normal exit (`try/finally` pattern from `Score.py`)
- [ ] Remove / archive `Score.py` and `Pinball_Camera.py` once their logic is fully absorbed
- [ ] Update `requirements.txt` if any new packages added

### Phase 8 — Arduino LED & Sound Integration *(deferred)*
- [ ] To be completed once Arduino sketches and communication method are finalized

---

## Verification Checklist

### Software (Simulation Mode — no hardware required)
- [ ] `python3 main.py` cycles through all states, printing transitions to console
- [ ] Mode select: keyboard input cycles options, simultaneous trigger confirms
- [ ] Drain event: simulated drain switch awards point to opponent, `POINT_SCORED` state shows updated scores briefly, returns to `GAMEPLAY`
- [ ] Win condition: score limit reached → `GAME_OVER` → winner display → auto-reset to attract
- [ ] Ctrl+C and normal exit: clean shutdown, no hanging threads

### Hardware (On Raspberry Pi)
- [ ] `i2cdetect -y 1` shows `0x26` and `0x27`
- [ ] `python3 display_test.py` — both LCDs initialize and cycle test patterns
- [ ] P1/P2 buttons trigger correct flipper relays (hold behavior verified)
- [ ] P1 drain switch awards a point to P2 (and vice versa); debounce prevents double-trigger
- [ ] `POINT_SCORED` state shows correct scorer and updated scores on LCD for 2s
- [ ] CV mode: camera thread launches, ball detected in both zones, correct relay fires with cooldown
- [ ] Human P2 mode: camera thread not running, P2 buttons drive relays
- [ ] Arduino signal fires on each game event (point scored, win, attract)
- [ ] All GPIO released cleanly on exit (no "GPIO in use" warnings on re-run)

