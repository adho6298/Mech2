#!/usr/bin/env python3
"""
================================================================================
PINBALL MACHINE — MAIN CONTROLLER
================================================================================

Single entry point for the 2-player pinball machine.

Gameplay: competitive, air-hockey style. Each player defends their side with
flippers. When the ball drains on your side, the OPPONENT scores a point.
First player to SCORE_LIMIT wins.

Player 2 mode is selected at game start:
  "1P vs CPU"  — P2 flippers driven automatically by camera (CV mode)
  "2P Human"   — P2 flippers driven by physical buttons

Run on Raspberry Pi 5:
    python3 main.py

Simulation mode (no hardware):
    Runs automatically when gpiozero/RPLCD are unavailable.
    Keyboard controls printed at startup.

See plan.md for full hardware wiring and GPIO pin assignments.
================================================================================
"""

import time
import threading
from enum import Enum

# ==============================================================================
# CONFIGURATION — edit these values to match your wiring
# ==============================================================================

# Scoring
SCORE_LIMIT = 3            # First to reach this wins — Arduino plays game-over animation at 3

# Timing
DEBOUNCE_TIME        = 0.2  # Seconds — drain switch software debounce
SIMULTANEOUS_WINDOW  = 0.1  # Seconds — window to detect "both P1 buttons at once"
POINT_SCORED_PAUSE   = 2.0  # Seconds — pause after each point before resuming
WIN_DISPLAY_TIME     = 5.0  # Seconds — winner screen duration before attract
COUNTDOWN_DELAY      = 1.0  # Seconds per countdown step

# LCD I2C addresses
LCD1_ADDR = 0x27            # Default address
LCD2_ADDR = 0x26            # Solder A0 jumper to set this address

# GPIO — drain switches (inputs, active-LOW with pull-up)
# Ball drains on P1's side  →  P2 scores
# Ball drains on P2's side  →  P1 scores
P1_DRAIN_PIN = 5            # Physical pin 29
P2_DRAIN_PIN = 6            # Physical pin 31

# GPIO — P1 (Red) flipper buttons (inputs, active-LOW with pull-up)
P1_LEFT_BTN_PIN  = 9        # Physical pin 21 — Red Left Button
P1_RIGHT_BTN_PIN = 16       # Physical pin 36 — Red Right Button (GPIO 14 avoided: UART TX conflict)

# GPIO — P1 (Red) flipper relays (outputs)
P1_LEFT_RELAY_PIN  = 11     # Physical pin 23 — Red Left Relay
P1_RIGHT_RELAY_PIN = 15     # Physical pin 10 — Red Right Relay

# GPIO — P2 (Blue) flipper relays (outputs) — used by CV mode AND human P2 mode
P2_LEFT_RELAY_PIN  = 27     # Physical pin 13 — Blue Left Relay
P2_RIGHT_RELAY_PIN = 10     # Physical pin 19 — Blue Right Relay

# GPIO — P2 (Blue) flipper buttons (inputs, active-LOW with pull-up) — human P2 mode only
P2_LEFT_BTN_PIN  = 17       # Physical pin 11 — Blue Left Button
P2_RIGHT_BTN_PIN = 22       # Physical pin 15 — Blue Right Button

# GPIO — Arduino signal pins (outputs, active-HIGH) — pulsed on game events
ARDUINO_P1_SCORE_PIN = 18   # Physical pin 12 — Red  player scored
ARDUINO_P2_SCORE_PIN = 23   # Physical pin 16 — Blue player scored
ARDUINO_START_PIN    = 24   # Physical pin 18 — Game start
ARDUINO_PULSE_TIME   = 0.1  # Seconds — how long the signal pin is held HIGH

# Camera detection tuning (from Pinball_Camera.py)
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 240
BRIGHTNESS_THRESHOLD = 180
CV_HIT_TIME          = 0.05  # Seconds — relay pulse duration per CV flipper hit
CV_EXTRA_COOLDOWN    = 0.4   # Seconds — additional cooldown after hit cycle

# ==============================================================================
# HARDWARE IMPORTS — graceful fallback to simulation mode
# ==============================================================================

try:
    from gpiozero import Button, OutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory
    from RPLCD.i2c import CharLCD
    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Hardware libraries unavailable: {e}")
    print("[WARNING] Running in simulation mode — all GPIO/LCD calls are stubbed.")
    HARDWARE_AVAILABLE = False

try:
    from picamera2 import Picamera2
    import cv2
    import numpy as np
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    # numpy stub so CameraThread code doesn't explode in simulation
    try:
        import numpy as np
    except ImportError:
        np = None

# ==============================================================================
# GAME STATE ENUM
# ==============================================================================

class GameState(Enum):
    ATTRACT      = "ATTRACT"
    MODE_SELECT  = "MODE_SELECT"
    GAME_START   = "GAME_START"
    GAMEPLAY     = "GAMEPLAY"
    POINT_SCORED = "POINT_SCORED"
    GAME_OVER    = "GAME_OVER"

# ==============================================================================
# GLOBAL STATE
# ==============================================================================

game_state   = GameState.ATTRACT
scores       = {"P1": 0, "P2": 0}
p2_mode      = "CV"          # "CV" or "HUMAN" — set at mode select
last_scorer  = None          # "P1" or "P2" — most recent point scorer
running      = True

# Drain switch debounce
last_drain_time = {"P1": 0.0, "P2": 0.0}

# Mode select
MODE_OPTIONS    = ["1P vs CPU", "2P Human"]
mode_select_idx = 0

# P1 button timestamps — used to detect simultaneous press
p1_btn_press_time = {"left": 0.0, "right": 0.0}

# Pending deferred cycle timer (cancelled if simultaneous press detected)
_pending_cycle_timer = None

# Hardware handles
lcds           = []
_gpio_factory  = None

p1_drain_btn   = None
p2_drain_btn   = None
p1_left_btn    = None
p1_right_btn   = None
p2_left_btn    = None
p2_right_btn   = None

p1_left_relay  = None
p1_right_relay = None
p2_left_relay  = None
p2_right_relay = None

arduino_p1_score_dev = None
arduino_p2_score_dev = None
arduino_start_dev    = None

camera_thread  = None

# ==============================================================================
# LCD FUNCTIONS
# ==============================================================================

def init_lcds() -> bool:
    global lcds
    if not HARDWARE_AVAILABLE:
        print("[SIM] LCDs initialised")
        return True
    try:
        lcd1 = CharLCD(i2c_expander='PCF8574', address=LCD1_ADDR, port=1,
                       cols=16, rows=2, dotsize=8, auto_linebreaks=True)
        lcd2 = CharLCD(i2c_expander='PCF8574', address=LCD2_ADDR, port=1,
                       cols=16, rows=2, dotsize=8, auto_linebreaks=True)
        lcds = [lcd1, lcd2]
        print(f"[OK] LCD1 at 0x{LCD1_ADDR:02X}, LCD2 at 0x{LCD2_ADDR:02X}")
        return True
    except Exception as e:
        print(f"[ERROR] LCD init failed: {e}")
        return False


def write_to_lcds(line1: str, line2: str):
    """Write identical content to both LCDs simultaneously."""
    line1 = line1[:16].ljust(16)
    line2 = line2[:16].ljust(16)
    if not HARDWARE_AVAILABLE:
        print(f"[LCD] \"{line1}\" | \"{line2}\"")
        return
    for lcd in lcds:
        try:
            lcd.home()
            lcd.write_string(line1)
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2)
        except Exception as e:
            print(f"[ERROR] LCD write failed: {e}")


def show_attract():
    write_to_lcds("  PINBALL 2026  ", " >> PRESS START ")


def show_mode_select(idx: int):
    # Show both options; selected one has a ">" prefix
    lines = [
        (">" if i == idx else " ") + " " + MODE_OPTIONS[i]
        for i in range(len(MODE_OPTIONS))
    ]
    write_to_lcds(lines[0], lines[1])


def show_countdown(n: int):
    write_to_lcds("  GET READY...  ", f"       {n}...     ")


def show_scores():
    p1 = str(scores["P1"])
    p2 = str(scores["P2"])
    write_to_lcds("P1            P2", p1.ljust(8) + p2.rjust(8))


def show_point_scored(scorer: str):
    line1 = f"  {scorer} SCORES!  "
    p1 = str(scores["P1"])
    p2 = str(scores["P2"])
    write_to_lcds(line1, p1.ljust(8) + p2.rjust(8))


def show_winner(winner: str):
    write_to_lcds(f" PLAYER {winner[-1]} WINS! ", ">" * 16)


def clear_lcds():
    if not HARDWARE_AVAILABLE:
        return
    for lcd in lcds:
        try:
            lcd.clear()
        except Exception:
            pass


def close_lcds():
    if not HARDWARE_AVAILABLE:
        return
    for lcd in lcds:
        try:
            lcd.clear()
            lcd.close(clear=True)
        except Exception:
            pass

# ==============================================================================
# ARDUINO SIGNALING
# ==============================================================================

def arduino_signal(event: str):
    """
    Pulses the appropriate Arduino signal pin HIGH for ARDUINO_PULSE_TIME seconds.
    The Arduino reads these as digitalRead() inputs.

    Signals sent by the RPi:
      'game_start' — pulse GPIO 24 once; Arduino begins game mode
      'p1_scored'  — pulse GPIO 18 once; Arduino increments P1 count
      'p2_scored'  — pulse GPIO 23 once; Arduino increments P2 count

    The Arduino tracks the score independently and plays the game-over
    animation automatically when either player reaches 3.
    """
    print(f"[ARDUINO] Event: {event}")

    if event == "game_start":
        device = arduino_start_dev
    elif event == "p1_scored":
        device = arduino_p1_score_dev
    elif event == "p2_scored":
        device = arduino_p2_score_dev
    else:
        return  # 'attract', 'p1_wins', 'p2_wins' — Arduino handles these autonomously

    if device is None:
        return

    def _pulse():
        try:
            device.on()
            time.sleep(ARDUINO_PULSE_TIME)
            device.off()
        except Exception as e:
            print(f"[ARDUINO] Pulse error: {e}")

    threading.Thread(target=_pulse, daemon=True).start()

# ==============================================================================
# RELAY HELPERS
# ==============================================================================

def _relay_on(relay):
    if relay is not None and HARDWARE_AVAILABLE:
        relay.on()


def _relay_off(relay):
    if relay is not None and HARDWARE_AVAILABLE:
        relay.off()


def _get_relay(player: str, side: str):
    if player == "P1":
        return p1_left_relay if side == "left" else p1_right_relay
    else:
        return p2_left_relay if side == "left" else p2_right_relay


def all_relays_off():
    for relay in (p1_left_relay, p1_right_relay, p2_left_relay, p2_right_relay):
        _relay_off(relay)

# ==============================================================================
# CAMERA THREAD — CV auto-flipper for P2 (1P vs CPU mode)
# ==============================================================================

class CameraThread:
    """
    Runs the ball detection loop from Pinball_Camera.py on a background thread.
    When the ball is detected in the left or right zone, the P2 relay for that
    side is pulsed for CV_HIT_TIME seconds.

    Call start() when entering GAMEPLAY in CV mode.
    Call pause() / resume() around POINT_SCORED.
    Call stop() on GAME_OVER or shutdown.
    """

    def __init__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._active = threading.Event()   # set = running, clear = paused
        self._stop   = threading.Event()
        self._left_cooldown_end  = 0.0
        self._right_cooldown_end = 0.0

    def start(self):
        self._active.set()
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        print("[CAM] Camera thread started")

    def pause(self):
        self._active.clear()
        print("[CAM] Camera thread paused")

    def resume(self):
        self._active.set()
        print("[CAM] Camera thread resumed")

    def stop(self):
        self._active.clear()
        self._stop.set()
        self._thread.join(timeout=3.0)
        print("[CAM] Camera thread stopped")

    def _pulse_relay(self, relay):
        """Pulse relay HIGH for CV_HIT_TIME — used for CV auto-flipping."""
        try:
            _relay_on(relay)
            time.sleep(CV_HIT_TIME)
        finally:
            _relay_off(relay)
            time.sleep(CV_HIT_TIME)

    def _run(self):
        if not CAMERA_AVAILABLE:
            print("[CAM] picamera2/OpenCV not available — camera thread idle")
            while not self._stop.is_set():
                time.sleep(0.1)
            return

        picam2 = Picamera2()
        cfg = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "YUV420"},
            buffer_count=2
        )
        picam2.configure(cfg)
        picam2.start()

        midpoint = FRAME_WIDTH // 2
        kernel   = np.ones((3, 3), np.uint8)

        try:
            while not self._stop.is_set():
                if not self._active.is_set():
                    time.sleep(0.05)
                    continue

                # Capture Y-plane only (grayscale, fastest path)
                buffer = picam2.capture_buffer("main")
                y_len  = FRAME_WIDTH * FRAME_HEIGHT
                frame  = np.frombuffer(buffer, dtype=np.uint8, count=y_len).reshape(
                    (FRAME_HEIGHT, FRAME_WIDTH)
                )

                mask         = np.where(frame > BRIGHTNESS_THRESHOLD, 255, 0).astype(np.uint8)
                mask         = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                column_sums  = np.sum(mask, axis=0)
                left_sums    = column_sums[:midpoint]
                right_sums   = column_sums[midpoint:]
                left_total   = int(left_sums.sum())
                right_total  = int(right_sums.sum())
                now          = time.time()

                if left_total > 0 and now >= self._left_cooldown_end:
                    print("[CAM] Ball LEFT — actuating P2 left flipper")
                    self._pulse_relay(p2_left_relay)
                    self._left_cooldown_end = time.time() + CV_EXTRA_COOLDOWN

                if right_total > 0 and now >= self._right_cooldown_end:
                    print("[CAM] Ball RIGHT — actuating P2 right flipper")
                    self._pulse_relay(p2_right_relay)
                    self._right_cooldown_end = time.time() + CV_EXTRA_COOLDOWN

        finally:
            picam2.stop()
            print("[CAM] Camera released")

# ==============================================================================
# INPUT CALLBACKS
# ==============================================================================

def on_drain(drained_side: str):
    """
    Called when a drain switch fires.
    drained_side = the player whose side the ball drained on.
    The OPPONENT scores the point.
    """
    global game_state, last_scorer

    if game_state != GameState.GAMEPLAY:
        return

    now = time.time()
    if now - last_drain_time[drained_side] < DEBOUNCE_TIME:
        print(f"[DEBOUNCE] {drained_side} drain ignored")
        return
    last_drain_time[drained_side] = now

    scorer = "P2" if drained_side == "P1" else "P1"
    scores[scorer] += 1
    last_scorer = scorer
    print(f"[SCORE] Ball drained on {drained_side}'s side → {scorer} scores! "
          f"(P1:{scores['P1']} P2:{scores['P2']})")

    game_state = GameState.POINT_SCORED


def on_p1_left_press():
    """Unified P1 left button handler — behaviour depends on current game state."""
    global game_state
    if game_state == GameState.ATTRACT:
        game_state = GameState.MODE_SELECT
    elif game_state == GameState.MODE_SELECT:
        p1_btn_press_time["left"] = time.time()
        _cancel_pending_cycle()
        if not _check_simultaneous_confirm():
            _schedule_cycle("left")
    elif game_state in (GameState.GAMEPLAY, GameState.POINT_SCORED):
        _relay_on(p1_left_relay)
        print("[FLIPPER] P1 left ON")


def on_p1_left_release():
    _relay_off(p1_left_relay)
    print("[FLIPPER] P1 left OFF")


def on_p1_right_press():
    """Unified P1 right button handler — behaviour depends on current game state."""
    global game_state
    if game_state == GameState.ATTRACT:
        game_state = GameState.MODE_SELECT
    elif game_state == GameState.MODE_SELECT:
        p1_btn_press_time["right"] = time.time()
        _cancel_pending_cycle()
        if not _check_simultaneous_confirm():
            _schedule_cycle("right")
    elif game_state in (GameState.GAMEPLAY, GameState.POINT_SCORED):
        _relay_on(p1_right_relay)
        print("[FLIPPER] P1 right ON")


def on_p1_right_release():
    _relay_off(p1_right_relay)
    print("[FLIPPER] P1 right OFF")


def on_p2_left_press():
    """P2 human flipper — only active when p2_mode is HUMAN."""
    if p2_mode == "HUMAN" and game_state in (GameState.GAMEPLAY, GameState.POINT_SCORED):
        _relay_on(p2_left_relay)
        print("[FLIPPER] P2 left ON")


def on_p2_left_release():
    if p2_mode == "HUMAN":
        _relay_off(p2_left_relay)
        print("[FLIPPER] P2 left OFF")


def on_p2_right_press():
    if p2_mode == "HUMAN" and game_state in (GameState.GAMEPLAY, GameState.POINT_SCORED):
        _relay_on(p2_right_relay)
        print("[FLIPPER] P2 right ON")


def on_p2_right_release():
    if p2_mode == "HUMAN":
        _relay_off(p2_right_relay)
        print("[FLIPPER] P2 right OFF")


# ==============================================================================
# MODE SELECT HELPERS
# ==============================================================================

def _cycle_mode(direction: str):
    """Advance or reverse through MODE_OPTIONS and update the display."""
    global mode_select_idx
    if direction == "right":
        mode_select_idx = (mode_select_idx + 1) % len(MODE_OPTIONS)
    else:
        mode_select_idx = (mode_select_idx - 1) % len(MODE_OPTIONS)
    show_mode_select(mode_select_idx)
    print(f"[MODE] Highlighted: {MODE_OPTIONS[mode_select_idx]}")


def _cancel_pending_cycle():
    """Cancel any deferred single-button cycle that hasn't fired yet."""
    global _pending_cycle_timer
    if _pending_cycle_timer is not None:
        _pending_cycle_timer.cancel()
        _pending_cycle_timer = None


def _schedule_cycle(direction: str):
    """Defer a mode cycle by SIMULTANEOUS_WINDOW so a second button press can cancel it."""
    global _pending_cycle_timer

    def _do_cycle():
        global _pending_cycle_timer
        _pending_cycle_timer = None
        if game_state == GameState.MODE_SELECT:
            _cycle_mode(direction)

    _pending_cycle_timer = threading.Timer(SIMULTANEOUS_WINDOW, _do_cycle)
    _pending_cycle_timer.start()


def _check_simultaneous_confirm():
    """
    Confirm mode selection if both P1 buttons were pressed within
    SIMULTANEOUS_WINDOW of each other.  Returns True if confirmed.
    """
    global game_state, p2_mode
    left_t  = p1_btn_press_time["left"]
    right_t = p1_btn_press_time["right"]
    if left_t > 0 and right_t > 0 and abs(left_t - right_t) <= SIMULTANEOUS_WINDOW:
        # Reset timestamps so this doesn't re-fire
        p1_btn_press_time["left"]  = 0.0
        p1_btn_press_time["right"] = 0.0
        selected = MODE_OPTIONS[mode_select_idx]
        p2_mode  = "CV" if selected == "1P vs CPU" else "HUMAN"
        print(f"[MODE] Confirmed: {selected}  (p2_mode={p2_mode})")
        game_state = GameState.GAME_START
        return True
    return False

# ==============================================================================
# GPIO INIT & CLEANUP
# ==============================================================================

def init_gpio() -> bool:
    global _gpio_factory
    global p1_drain_btn, p2_drain_btn
    global p1_left_btn, p1_right_btn
    global p2_left_btn, p2_right_btn
    global p1_left_relay, p1_right_relay
    global p2_left_relay, p2_right_relay
    global arduino_p1_score_dev, arduino_p2_score_dev, arduino_start_dev

    if not HARDWARE_AVAILABLE:
        print("[SIM] GPIO initialised (keyboard simulation active)")
        return True

    try:
        _gpio_factory = LGPIOFactory()

        # Drain switches
        if P1_DRAIN_PIN is not None:
            p1_drain_btn = Button(P1_DRAIN_PIN, pull_up=True, bounce_time=0.05,
                                  pin_factory=_gpio_factory)
            p1_drain_btn.when_pressed = lambda: on_drain("P1")
            print(f"[OK] P1 drain switch: GPIO{P1_DRAIN_PIN}")
        else:
            print("[WARN] P1 drain switch not configured — set P1_DRAIN_PIN")

        if P2_DRAIN_PIN is not None:
            p2_drain_btn = Button(P2_DRAIN_PIN, pull_up=True, bounce_time=0.05,
                                  pin_factory=_gpio_factory)
            p2_drain_btn.when_pressed = lambda: on_drain("P2")
            print(f"[OK] P2 drain switch: GPIO{P2_DRAIN_PIN}")
        else:
            print("[WARN] P2 drain switch not configured — set P2_DRAIN_PIN")

        # P2 flipper relays (always present)
        p2_left_relay  = OutputDevice(P2_LEFT_RELAY_PIN,  initial_value=False,
                                      pin_factory=_gpio_factory)
        p2_right_relay = OutputDevice(P2_RIGHT_RELAY_PIN, initial_value=False,
                                      pin_factory=_gpio_factory)
        print(f"[OK] P2 relays: LEFT=GPIO{P2_LEFT_RELAY_PIN}, "
              f"RIGHT=GPIO{P2_RIGHT_RELAY_PIN}")

        # P1 flipper buttons (optional until pins are confirmed)
        if P1_LEFT_BTN_PIN is not None and P1_RIGHT_BTN_PIN is not None:
            p1_left_btn  = Button(P1_LEFT_BTN_PIN,  pull_up=False, bounce_time=0.02,
                                  pin_factory=_gpio_factory)
            p1_right_btn = Button(P1_RIGHT_BTN_PIN, pull_up=False, bounce_time=0.02,
                                  pin_factory=_gpio_factory)
            p1_left_btn.when_pressed   = on_p1_left_press
            p1_left_btn.when_released  = on_p1_left_release
            p1_right_btn.when_pressed  = on_p1_right_press
            p1_right_btn.when_released = on_p1_right_release
            print(f"[OK] P1 buttons: LEFT=GPIO{P1_LEFT_BTN_PIN}, "
                  f"RIGHT=GPIO{P1_RIGHT_BTN_PIN}")
        else:
            #hi
            print("[WARN] P1 button GPIO pins not set — update P1_LEFT/RIGHT_BTN_PIN "
                  "in the config block")

        # P1 flipper relays (optional until pins are confirmed)
        if P1_LEFT_RELAY_PIN is not None and P1_RIGHT_RELAY_PIN is not None:
            p1_left_relay  = OutputDevice(P1_LEFT_RELAY_PIN,  initial_value=False,
                                          pin_factory=_gpio_factory)
            p1_right_relay = OutputDevice(P1_RIGHT_RELAY_PIN, initial_value=False,
                                          pin_factory=_gpio_factory)
            print(f"[OK] P1 relays: LEFT=GPIO{P1_LEFT_RELAY_PIN}, "
                  f"RIGHT=GPIO{P1_RIGHT_RELAY_PIN}")
        else:
            print("[WARN] P1 relay GPIO pins not set — update P1_LEFT/RIGHT_RELAY_PIN "
                  "in the config block")

        # P2 human flipper buttons (optional)
        if P2_LEFT_BTN_PIN is not None and P2_RIGHT_BTN_PIN is not None:
            p2_left_btn  = Button(P2_LEFT_BTN_PIN,  pull_up=False, bounce_time=0.02,
                                  pin_factory=_gpio_factory)
            p2_right_btn = Button(P2_RIGHT_BTN_PIN, pull_up=False, bounce_time=0.02,
                                  pin_factory=_gpio_factory)
            p2_left_btn.when_pressed   = on_p2_left_press
            p2_left_btn.when_released  = on_p2_left_release
            p2_right_btn.when_pressed  = on_p2_right_press
            p2_right_btn.when_released = on_p2_right_release
            print(f"[OK] P2 buttons: LEFT=GPIO{P2_LEFT_BTN_PIN}, "
                  f"RIGHT=GPIO{P2_RIGHT_BTN_PIN}")
        else:
            print("[WARN] P2 button GPIO pins not set — update P2_LEFT/RIGHT_BTN_PIN "
                  "in the config block")

        # Arduino signal output pins
        arduino_p1_score_dev = OutputDevice(ARDUINO_P1_SCORE_PIN, initial_value=False,
                                            pin_factory=_gpio_factory)
        arduino_p2_score_dev = OutputDevice(ARDUINO_P2_SCORE_PIN, initial_value=False,
                                            pin_factory=_gpio_factory)
        arduino_start_dev    = OutputDevice(ARDUINO_START_PIN,    initial_value=False,
                                            pin_factory=_gpio_factory)
        print(f"[OK] Arduino signals: P1_SCORE=GPIO{ARDUINO_P1_SCORE_PIN}, "
              f"P2_SCORE=GPIO{ARDUINO_P2_SCORE_PIN}, START=GPIO{ARDUINO_START_PIN}")

        return True

    except Exception as e:
        print(f"[ERROR] GPIO init failed: {e}")
        return False


def cleanup_gpio():
    """Release all GPIO resources cleanly."""
    all_relays_off()
    if not HARDWARE_AVAILABLE:
        return
    for device in (p1_drain_btn, p2_drain_btn,
                   p1_left_btn, p1_right_btn,
                   p2_left_btn, p2_right_btn,
                   p1_left_relay, p1_right_relay,
                   p2_left_relay, p2_right_relay,
                   arduino_p1_score_dev, arduino_p2_score_dev, arduino_start_dev):
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
    print("[GPIO] Cleanup complete")

# ==============================================================================
# SIMULATION KEYBOARD INPUT (non-hardware testing)
# ==============================================================================

def _start_sim_keyboard():
    """
    Start a pynput keyboard listener that mirrors hardware input for testing.

    Controls:
        ATTRACT     : any key            → MODE_SELECT
        MODE_SELECT : A / D              → cycle left / right
                      SPACE or ENTER    → confirm selection
        GAMEPLAY    : 1                  → P1 drains (P2 scores)
                      2                  → P2 drains (P1 scores)
                      Z / X             → P1 left / right flipper
                      N / M             → P2 left / right flipper
        ANY STATE   : Q                  → quit
    """
    try:
        from pynput import keyboard as kb
    except ImportError:
        print("[WARN] pynput not available — keyboard simulation disabled")
        return None

    def on_press(key):
        global game_state, running
        try:
            ch = key.char
        except AttributeError:
            ch = None

        if ch == 'q':
            running = False
            return

        if game_state == GameState.ATTRACT:
            if ch is not None:
                game_state = GameState.MODE_SELECT
            return

        if game_state == GameState.MODE_SELECT:
            if ch == 'a':
                p1_btn_press_time["left"] = time.time()
                _cycle_mode("left")
            elif ch == 'd':
                p1_btn_press_time["right"] = time.time()
                _cycle_mode("right")
            elif ch in (' ', '\r'):
                # Simulate simultaneous press by zeroing the delta
                _cancel_pending_cycle()
                p1_btn_press_time["left"]  = time.time()
                p1_btn_press_time["right"] = time.time()
                _check_simultaneous_confirm()
            return

        if game_state == GameState.GAMEPLAY:
            if ch == '1':
                on_drain("P1")
            elif ch == '2':
                on_drain("P2")
            elif ch == 'z':
                on_p1_left_press()
            elif ch == 'x':
                on_p1_right_press()
            elif ch == 'n':
                on_p2_left_press()
            elif ch == 'm':
                on_p2_right_press()

    def on_release(key):
        try:
            ch = key.char
        except AttributeError:
            ch = None
        if ch == 'z':
            on_p1_left_release()
        elif ch == 'x':
            on_p1_right_release()
        elif ch == 'n':
            on_p2_left_release()
        elif ch == 'm':
            on_p2_right_release()

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    print("[SIM] Keyboard controls:")
    print("  ATTRACT     : any key          → MODE_SELECT")
    print("  MODE_SELECT : A / D            → cycle options")
    print("                SPACE / ENTER   → confirm selection")
    print("  GAMEPLAY    : 1               → P1 drains (P2 scores)")
    print("                2               → P2 drains (P1 scores)")
    print("                Z / X          → P1 left / right flipper")
    print("                N / M          → P2 left / right flipper")
    print("  ANY STATE   : Q               → quit")
    return listener

# ==============================================================================
# MAIN RUN LOOP — state machine
# ==============================================================================

def run():
    global game_state, running, mode_select_idx, camera_thread

    print("=" * 52)
    print("   PINBALL MACHINE — MAIN CONTROLLER")
    print(f"   Score Limit : {SCORE_LIMIT if SCORE_LIMIT > 0 else 'unlimited'}")
    print(f"   Hardware    : {'YES' if HARDWARE_AVAILABLE else 'SIMULATION MODE'}")
    print(f"   Camera      : {'YES' if CAMERA_AVAILABLE else 'NOT AVAILABLE'}")
    print("=" * 52)

    # Initialise subsystems
    if not init_lcds():
        print("[ERROR] LCD init failed — display will be console only")

    if not init_gpio():
        print("[ERROR] GPIO init failed — inputs/outputs may not work")

    camera_thread = CameraThread()

    sim_listener = None
    if not HARDWARE_AVAILABLE:
        sim_listener = _start_sim_keyboard()

    # Kick off in ATTRACT
    game_state = GameState.ATTRACT
    prev_state = None

    try:
        while running:
            current = game_state

            # -----------------------------------------------------------------
            # State entry actions — run once each time we enter a new state
            # -----------------------------------------------------------------
            if current != prev_state:
                prev_state = current
                print(f"[STATE] → {current.value}")

                if current == GameState.ATTRACT:
                    show_attract()
                    arduino_signal("attract")

                elif current == GameState.MODE_SELECT:
                    mode_select_idx = 0
                    p1_btn_press_time["left"]  = 0.0
                    p1_btn_press_time["right"] = 0.0
                    show_mode_select(0)

                elif current == GameState.GAME_START:
                    # Countdown then auto-transition to GAMEPLAY
                    all_relays_off()
                    for n in (3, 2, 1):
                        show_countdown(n)
                        time.sleep(COUNTDOWN_DELAY)
                    arduino_signal("game_start")
                    game_state = GameState.GAMEPLAY

                elif current == GameState.GAMEPLAY:
                    show_scores()
                    if p2_mode == "CV":
                        camera_thread.start()

                elif current == GameState.POINT_SCORED:
                    if p2_mode == "CV":
                        camera_thread.pause()
                    show_point_scored(last_scorer)
                    arduino_signal(f"{'p1' if last_scorer == 'P1' else 'p2'}_scored")

                elif current == GameState.GAME_OVER:
                    if p2_mode == "CV":
                        camera_thread.stop()
                    all_relays_off()
                    winner = last_scorer
                    # No signal sent here — Arduino counts to 3 and plays game-over animation itself
                    # Flash winner message
                    for _ in range(5):
                        show_winner(winner)
                        time.sleep(WIN_DISPLAY_TIME / 10)
                        clear_lcds()
                        time.sleep(WIN_DISPLAY_TIME / 10)
                    # Reset scores and return to attract
                    scores["P1"] = 0
                    scores["P2"] = 0
                    game_state = GameState.ATTRACT

            # -----------------------------------------------------------------
            # State-specific continuous logic
            # -----------------------------------------------------------------
            elif current == GameState.POINT_SCORED:
                # Hold the POINT_SCORED display for the configured pause then resume
                time.sleep(POINT_SCORED_PAUSE)
                if SCORE_LIMIT > 0 and scores.get(last_scorer, 0) >= SCORE_LIMIT:
                    game_state = GameState.GAME_OVER
                else:
                    show_scores()
                    game_state = GameState.GAMEPLAY
                prev_state = None  # Force re-entry of next state

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt — shutting down")

    finally:
        running = False
        if camera_thread is not None:
            try:
                camera_thread.stop()
            except Exception:
                pass
        cleanup_gpio()
        close_lcds()
        if sim_listener is not None:
            try:
                sim_listener.stop()
            except Exception:
                pass
        print("[MAIN] Shutdown complete")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    run()
