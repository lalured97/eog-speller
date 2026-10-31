// EOG stream for Arduino UNO Q (Team B, step B6)
// Reads ADS1115 AIN0 (H channel) and AIN1 (V channel) 250 times a second
// and sends "h,v" lines (raw ADC counts) to the PC over USB.
//
// UNO Q notes:
//  - Serial on the UNO Q goes to pins D0/D1, NOT to USB.
//    To reach the PC over the USB cable we use Monitor (Arduino_RouterBridge).
//  - Header SDA/SCL pins use "Wire". The Qwiic connector would be "Wire1".
//  - The UNO Q runs at 3.3 V: power the ADS1115 from 3.3 V.
// No ADS1115 library is used: we talk to its registers directly with Wire,
// so there is nothing extra to install and you can see exactly what happens.

#include <Wire.h>
#include <Arduino_RouterBridge.h>

const uint8_t ADS_ADDR = 0x48;          // ADDR pin to GND
const uint8_t REG_CONV = 0x00;
const uint8_t REG_CONFIG = 0x01;
// Config = start single conversion | input | +/-4.096 V range | single-shot
//          | 860 samples/s | comparator off
const uint16_t CFG_AIN0 = 0xC3E3;       // AIN0 vs GND
const uint16_t CFG_AIN1 = 0xD3E3;       // AIN1 vs GND
const unsigned long PERIOD_US = 4000;   // 250 Hz

void writeReg(uint8_t reg, uint16_t value) {
  Wire.beginTransmission(ADS_ADDR);
  Wire.write(reg);
  Wire.write(value >> 8);
  Wire.write(value & 0xFF);
  Wire.endTransmission();
}

uint16_t readReg(uint8_t reg) {
  Wire.beginTransmission(ADS_ADDR);
  Wire.write(reg);
  Wire.endTransmission();
  Wire.requestFrom(ADS_ADDR, (uint8_t)2);
  uint16_t hi = Wire.read();
  uint16_t lo = Wire.read();
  return (hi << 8) | lo;
}

int16_t readChannel(uint16_t cfg) {
  writeReg(REG_CONFIG, cfg);                 // start a conversion
  delayMicroseconds(1200);                   // one conversion at 860 SPS
  while ((readReg(REG_CONFIG) & 0x8000) == 0) { }   // wait until done
  return (int16_t)readReg(REG_CONV);
}

unsigned long nextT;

void setup() {
  Bridge.begin();
  Monitor.begin();
  Wire.begin();
  Wire.setClock(400000);                     // fast I2C, needed for 250 Hz x 2
  Wire.beginTransmission(ADS_ADDR);
  if (Wire.endTransmission() != 0) {
    while (true) { Monitor.println("ADS1115 not found - check wiring"); delay(1000); }
  }
  nextT = micros();
}

void loop() {
  if ((long)(micros() - nextT) < 0) return;  // wait for the next 4 ms slot
  nextT += PERIOD_US;
  int16_t h = readChannel(CFG_AIN0);
  int16_t v = readChannel(CFG_AIN1);
  Monitor.print(h);
  Monitor.print(",");
  Monitor.println(v);
}
