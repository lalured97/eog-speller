"""Shared settings. All three teams agree on these numbers.
Change a value here only after telling the other teams."""

FS = 250                 # samples per second, per channel
BAUD = 115200            # Arduino -> PC serial speed

# Analog front end (Pair A). Used to convert ADC counts back to microvolts.
VREF = 1.65              # mid-supply bias voltage (V)
ANALOG_GAIN = 508        # INA333 (5.03) x OPA2333 stage (101). Pair A updates after measuring.
ADC_LSB = 0.000125       # ADS1115 at GAIN_ONE: 125 uV per count


def counts_to_uV(counts):
    """ADS1115 counts -> microvolts at the electrodes."""
    volts = counts * ADC_LSB
    return (volts - VREF) / ANALOG_GAIN * 1e6


# Event names sent from the recogniser to the speller (both Pair C)
EVENTS = ["LEFT", "RIGHT", "UP", "DOWN", "BLINK", "DOUBLE_BLINK"]
