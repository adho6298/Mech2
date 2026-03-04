#!/usr/bin/env python3
"""
================================================================================
LCD DISPLAY TEST - Verify I2C LCD 1602 connections
================================================================================

DESCRIPTION:
    Simple test script to verify both LCD displays are working before
    running the full Score.py scoreboard system.

HARDWARE SETUP:
    - Raspberry Pi 5 with I2C enabled
    - 2x FREENOVE I2C LCD 1602 Modules
        - LCD 1: Address 0x27 (default)
        - LCD 2: Address 0x26 (solder A0 jumper)

WIRING DIAGRAM:
    LCD Connections (both LCDs share same pins):
        VCC  → Pin 2 (5V)
        GND  → Pin 6 (GND)
        SDA  → Pin 3 (GPIO2)
        SCL  → Pin 5 (GPIO3)

PRE-FLIGHT CHECK:
    sudo raspi-config  # Enable I2C under Interface Options
    sudo apt install i2c-tools
    i2cdetect -y 1     # Should show 0x26 and 0x27

RUNNING:
    cd Pinball_Machine
    pip install -r requirements.txt
    python3 display_test.py

EXPECTED OUTPUT:
    Both LCDs will cycle through test patterns:
    1. "LCD TEST" / "Starting..."
    2. Each LCD shows its address
    3. Counter increments on both displays
    Press Ctrl+C to exit.

================================================================================
"""

import time

# =============================================================================
# CONFIGURATION - Must match Score.py
# =============================================================================

# LCD I2C Addresses (run 'i2cdetect -y 1' to verify)
LCD1_ADDR = 0x27          # Default address
LCD2_ADDR = 0x26          # Modified address (A0 jumper soldered)

# =============================================================================
# LCD INITIALIZATION
# =============================================================================

try:
    from RPLCD.i2c import CharLCD
    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] RPLCD library not available: {e}")
    print("[ERROR] Install with: pip install RPLCD")
    HARDWARE_AVAILABLE = False

lcds = []


def init_lcd(address, name):
    """Initialize a single LCD and return it."""
    try:
        lcd = CharLCD(
            i2c_expander='PCF8574',
            address=address,
            port=1,
            cols=16,
            rows=2,
            dotsize=8,
            auto_linebreaks=True
        )
        print(f"[OK] {name} initialized at address 0x{address:02X}")
        return lcd
    except Exception as e:
        print(f"[ERROR] {name} failed at 0x{address:02X}: {e}")
        return None


def write_lcd(lcd, line1, line2):
    """Write two lines to a single LCD."""
    if lcd is None:
        return
    try:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1[:16].ljust(16))
        lcd.cursor_pos = (1, 0)
        lcd.write_string(line2[:16].ljust(16))
    except Exception as e:
        print(f"[ERROR] Write failed: {e}")


def write_both(line1, line2):
    """Write the same content to both LCDs."""
    for lcd in lcds:
        write_lcd(lcd, line1, line2)


# =============================================================================
# MAIN TEST SEQUENCE
# =============================================================================

def main():
    global lcds
    
    print("=" * 50)
    print("LCD DISPLAY TEST")
    print("=" * 50)
    
    if not HARDWARE_AVAILABLE:
        print("[ERROR] Cannot run test without RPLCD library")
        return
    
    # Initialize LCDs
    print("\n[TEST 1] Initializing LCDs...")
    lcd1 = init_lcd(LCD1_ADDR, "LCD 1")
    lcd2 = init_lcd(LCD2_ADDR, "LCD 2")
    
    lcds = [lcd for lcd in [lcd1, lcd2] if lcd is not None]
    
    if len(lcds) == 0:
        print("[ERROR] No LCDs detected. Check wiring and I2C addresses.")
        print("[TIP] Run 'i2cdetect -y 1' to scan for devices.")
        return
    
    print(f"\n[INFO] {len(lcds)} LCD(s) detected")
    
    # Test 2: Show startup message on both
    print("\n[TEST 2] Displaying startup message...")
    write_both("LCD TEST", "Starting...")
    time.sleep(2)
    
    # Test 3: Show address on each LCD individually
    print("\n[TEST 3] Showing individual addresses...")
    if lcd1:
        write_lcd(lcd1, "LCD 1", f"Addr: 0x{LCD1_ADDR:02X}")
    if lcd2:
        write_lcd(lcd2, "LCD 2", f"Addr: 0x{LCD2_ADDR:02X}")
    time.sleep(3)
    
    # Test 4: Synchronized counter
    print("\n[TEST 4] Running synchronized counter (Ctrl+C to stop)...")
    try:
        counter = 0
        while True:
            write_both("Count Test:", f"{counter:^16}")
            print(f"[COUNTER] {counter}")
            counter += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Test interrupted")
    
    # Cleanup
    print("\n[CLEANUP] Clearing displays...")
    for lcd in lcds:
        try:
            lcd.clear()
            lcd.close(clear=True)
        except:
            pass
    
    print("[DONE] Display test complete")


if __name__ == "__main__":
    main()
