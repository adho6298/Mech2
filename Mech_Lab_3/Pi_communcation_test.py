import serial
import re
import time

baud_rate = 9600
ser = serial.Serial('/dev/ttyACM0',baud_rate)


while True:

    amount = int(input("Type in a random number"))

    byte = amount.to_bytes(1, byteorder='big', signed=False)
    ser.write(byte)

    ser.write(byte) #sending this guy


