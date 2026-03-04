#!/usr/bin/env python3
"""
================================================================================
PINBALL SCOREBOARD SYSTEM - HARDWARE TEST INSTRUCTIONS
================================================================================

DESCRIPTION:
    Tracks and displays scores for a two-player pinball game using dual LCD
    displays. Ender 3 limit switches detect scoring events with debounce
    protection. Displays synchronized output on two I2C LCD 1602 modules.

HARDWARE SETUP:
    - Raspberry Pi 5 with I2C enabled
    - 2x FREENOVE I2C LCD 1602 Modules
        - LCD 1: Address 0x27 (default)
        - LCD 2: Address 0x26 (solder A0 jumper)
    - 2x Ender 3 limit switches (NO - normally open)
        - P1 switch: GPIO 22 (Physical pin 15)
        - P2 switch: GPIO 23 (Physical pin 16)
        - Connect COM to GND, NO to GPIO pin
        - NOTE: GPIO 17/27 reserved for Pinball_Camera.py relays

WIRING DIAGRAM:
    LCD Connections (both LCDs share same pins):
        VCC  → Pin 2 (5V)
        GND  → Pin 6 (GND)
        SDA  → Pin 3 (GPIO2)
        SCL  → Pin 5 (GPIO3)

    Switch Connections:
        P1 Switch: COM → GND (Pin 14), NO → GPIO22 (Pin 15)
        P2 Switch: COM → GND (Pin 14), NO → GPIO23 (Pin 16)

INSTALLATION:
    cd Pinball_Machine
    pip install -r requirements.txt
    python3 "Score.py"

VERIFYING I2C ADDRESSES:
    sudo raspi-config  # Enable I2C under Interface Options
    sudo apt install i2c-tools
    i2cdetect -y 1     # Should show 0x26 and 0x27

DISPLAY FORMAT (16x2 LCD):
    Line 1: "P1            P2"
    Line 2: "5             12"  (scores left/right aligned)

CONTROLS:
    - P1 limit switch: Add point to Player 1
    - P2 limit switch: Add point to Player 2
    - Keyboard 'r': Reset scores to 0-0
    - Keyboard 'q': Quit program
    - Ctrl+C: Emergency shutdown

SIMULATION MODE:
    When hardware libraries are unavailable, the program runs in simulation:
    - Press '1' to simulate P1 score
    - Press '2' to simulate P2 score
    - LCD output prints to terminal

TUNING:
    Adjust variables in the CONFIGURATION section below:
    - SCORE_LIMIT: Points needed to win (0 = no limit)
    - DEBOUNCE_TIME: Seconds to ignore repeated triggers (default: 0.2)
    - WIN_DISPLAY_TIME: Seconds to show winner message (default: 5)
    - LCD1_ADDR/LCD2_ADDR: I2C addresses (verify with i2cdetect)
    - P1_SWITCH_PIN/P2_SWITCH_PIN: GPIO pins (BCM numbering)

TROUBLESHOOTING:
    - LCD not detected? Run 'i2cdetect -y 1' to verify addresses
    - Switches not working? Check COM→GND and NO→GPIO wiring
    - Double scoring? Increase DEBOUNCE_TIME
    - Pi 5 GPIO errors? Ensure lgpio is installed (in requirements.txt)
    - pynput errors? Run with 'sudo' or check X11 display access

================================================================================
"""

import time
import threading
from signal import signal, SIGINT

# =============================================================================
# CONFIGURATION - Adjust these values as needed
# =============================================================================

SCORE_LIMIT = 10          # First player to reach this score wins (0 = no limit)
DEBOUNCE_TIME = 0.2       # Seconds to ignore repeated switch triggers
WIN_DISPLAY_TIME = 5      # Seconds to show winner message before reset

# LCD I2C Addresses (run 'i2cdetect -y 1' to verify)
LCD1_ADDR = 0x27          # Default address
LCD2_ADDR = 0x26          # Modified address (A0 jumper soldered)

# GPIO Pin Numbers (BCM numbering)
P1_SWITCH_PIN = 22        # Physical pin 15
P2_SWITCH_PIN = 23        # Physical pin 16

# =============================================================================
# IMPORTS - Hardware libraries (only work on Raspberry Pi)
# =============================================================================

try:
    from gpiozero import Button
    from gpiozero.pins.lgpio import LGPIOFactory
    from RPLCD.i2c import CharLCD
    from pynput import keyboard
    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Hardware libraries not available: {e}")
    print("[WARNING] Running in simulation mode.")
    HARDWARE_AVAILABLE = False

# =============================================================================
# GLOBAL STATE
# =============================================================================

scores = {"P1": 0, "P2": 0}
last_trigger_time = {"P1": 0, "P2": 0}
game_active = True
running = True
lcds = []

# =============================================================================
# LCD FUNCTIONS
# =============================================================================

def init_lcds():
    """Initialize both LCD displays."""
    global lcds
    if not HARDWARE_AVAILABLE:
        print("[SIM] LCDs initialized")
        return True
    
    try:
        lcd1 = CharLCD(
            i2c_expander='PCF8574',
            address=LCD1_ADDR,
            port=1,
            cols=16,
            rows=2,
            dotsize=8,
            auto_linebreaks=True
        )
        lcd2 = CharLCD(
            i2c_expander='PCF8574',
            address=LCD2_ADDR,
            port=1,
            cols=16,
            rows=2,
            dotsize=8,
            auto_linebreaks=True
        )
        lcds = [lcd1, lcd2]
        print(f"[OK] LCD 1 initialized at address 0x{LCD1_ADDR:02X}")
        print(f"[OK] LCD 2 initialized at address 0x{LCD2_ADDR:02X}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize LCDs: {e}")
        return False


def write_to_lcds(line1: str, line2: str):
    """Write the same content to both LCDs simultaneously."""
    # Ensure lines are exactly 16 characters
    line1 = line1[:16].ljust(16)
    line2 = line2[:16].ljust(16)
    
    if not HARDWARE_AVAILABLE:
        print(f"[LCD] {line1}")
        print(f"[LCD] {line2}")
        print("-" * 20)
        return
    
    for lcd in lcds:
        try:
            lcd.clear()
            lcd.cursor_pos = (0, 0)
            lcd.write_string(line1)
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2)
        except Exception as e:
            print(f"[ERROR] LCD write failed: {e}")


def update_display():
    """Update LCDs with current scores."""
    # Format: "P1            P2" on line 1
    # Format: "5             12" on line 2 (scores under player names)
    line1 = "P1            P2"
    
    # Format scores: P1 left-aligned (positions 0-7), P2 right-aligned (positions 8-15)
    p1_score = str(scores["P1"])
    p2_score = str(scores["P2"])
    
    # Build line 2: score1 on left, score2 on right
    line2 = p1_score.ljust(8) + p2_score.rjust(8)
    
    write_to_lcds(line1, line2)


def display_winner(winner: str):
    """Flash winner message on LCDs."""
    win_msg = f"PLAYER {winner[-1]} WINS!"
    
    for i in range(5):  # Flash 5 times over WIN_DISPLAY_TIME
        write_to_lcds(win_msg.center(16), ">" * 16)
        time.sleep(WIN_DISPLAY_TIME / 10)
        write_to_lcds("", "")
        time.sleep(WIN_DISPLAY_TIME / 10)


# =============================================================================
# GAME LOGIC
# =============================================================================

def reset_scores():
    """Reset both scores to zero."""
    global game_active
    scores["P1"] = 0
    scores["P2"] = 0
    game_active = True
    print("[GAME] Scores reset to 0-0")
    update_display()


def add_score(player: str):
    """Add a point to a player with debounce check."""
    global game_active
    
    if not game_active:
        return
    
    current_time = time.time()
    
    # Debounce check
    if current_time - last_trigger_time[player] < DEBOUNCE_TIME:
        print(f"[DEBOUNCE] {player} trigger ignored (too fast)")
        return
    
    last_trigger_time[player] = current_time
    scores[player] += 1
    print(f"[SCORE] {player}: {scores[player]}")
    
    # Check win condition
    if SCORE_LIMIT > 0 and scores[player] >= SCORE_LIMIT:
        game_active = False
        print(f"[GAME] {player} wins!")
        display_winner(player)
        reset_scores()
    else:
        update_display()


# =============================================================================
# GPIO SWITCH HANDLERS
# =============================================================================

def on_p1_switch():
    """Callback for Player 1 switch press."""
    add_score("P1")


def on_p2_switch():
    """Callback for Player 2 switch press."""
    add_score("P2")


def init_switches():
    """Initialize GPIO switches with internal pull-ups."""
    if not HARDWARE_AVAILABLE:
        print("[SIM] Switches initialized (use keyboard 1/2 to simulate)")
        return None, None
    
    try:
        # Use lgpio factory for Pi 5 compatibility
        factory = LGPIOFactory()
        
        # Create buttons with pull-up resistors
        # bounce_time handles hardware debounce, we also do software debounce
        p1_button = Button(
            P1_SWITCH_PIN,
            pull_up=True,
            bounce_time=0.05,
            pin_factory=factory
        )
        p2_button = Button(
            P2_SWITCH_PIN,
            pull_up=True,
            bounce_time=0.05,
            pin_factory=factory
        )
        
        # Attach callbacks
        p1_button.when_pressed = on_p1_switch
        p2_button.when_pressed = on_p2_switch
        
        print(f"[OK] P1 switch on GPIO{P1_SWITCH_PIN}")
        print(f"[OK] P2 switch on GPIO{P2_SWITCH_PIN}")
        
        return p1_button, p2_button
    except Exception as e:
        print(f"[ERROR] Failed to initialize switches: {e}")
        return None, None


# =============================================================================
# KEYBOARD INPUT (for reset and simulation)
# =============================================================================

def on_key_press(key):
    """Handle keyboard input."""
    global running
    
    try:
        if hasattr(key, 'char'):
            if key.char == 'r':
                print("[INPUT] Reset triggered")
                reset_scores()
            elif key.char == '1' and not HARDWARE_AVAILABLE:
                on_p1_switch()
            elif key.char == '2' and not HARDWARE_AVAILABLE:
                on_p2_switch()
            elif key.char == 'q':
                print("[INPUT] Quit triggered")
                running = False
                return False  # Stop listener
    except AttributeError:
        pass


def start_keyboard_listener():
    """Start non-blocking keyboard listener."""
    if not HARDWARE_AVAILABLE:
        print("[SIM] Press 1/2 to score, 'r' to reset, 'q' to quit")
    else:
        print("[INPUT] Press 'r' to reset, 'q' to quit")
    
    try:
        listener = keyboard.Listener(on_press=on_key_press)
        listener.start()
        return listener
    except Exception as e:
        print(f"[WARNING] Keyboard listener failed: {e}")
        print("[WARNING] Manual reset via keyboard disabled")
        return None


# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

def shutdown(signum, frame):
    """Graceful shutdown handler."""
    global running
    print("\n[SHUTDOWN] Cleaning up...")
    running = False
    
    # Clear LCDs
    if HARDWARE_AVAILABLE and lcds:
        for lcd in lcds:
            try:
                lcd.clear()
                lcd.close(clear=True)
            except:
                pass
    
    print("[SHUTDOWN] Goodbye!")
    exit(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    global running
    
    print("=" * 50)
    print("PINBALL SCOREBOARD SYSTEM")
    print(f"Score Limit: {SCORE_LIMIT if SCORE_LIMIT > 0 else 'None'}")
    print(f"Debounce Time: {DEBOUNCE_TIME}s")
    print("=" * 50)
    
    # Register signal handler for Ctrl+C
    signal(SIGINT, shutdown)
    
    # Initialize hardware
    if not init_lcds():
        print("[ERROR] LCD initialization failed. Check I2C connections.")
        if HARDWARE_AVAILABLE:
            return
    
    switches = init_switches()
    keyboard_listener = start_keyboard_listener()
    
    # Initial display
    reset_scores()
    
    print("\n[READY] Scoreboard running. Waiting for input...")
    
    # Main loop
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(None, None)


if __name__ == "__main__":
    main()
