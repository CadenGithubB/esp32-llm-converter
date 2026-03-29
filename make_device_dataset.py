#!/usr/bin/env python3
"""
make_device_dataset.py — generate a Hardware One Q&A fine-tuning corpus.

Produces hardwareone_qa.txt: a plain-text file of short conversational
exchanges about the device, its commands, sensors, and features.  Feed it
to the training script with --text hardwareone_qa.txt --finetune-from.

Usage:
    python3 make_device_dataset.py [--out hardwareone_qa.txt] [--repeat 4]

  --repeat N   How many times to cycle through all pairs (default 4).
               More repetitions = model memorises the domain better,
               but too many = overfitting.  4-6 is a good range.
"""

import argparse
import random
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Q&A PAIRS
# Each entry is (question_variants, answer).
# Multiple question variants give the model exposure to different phrasings.
# Answers are kept short (1-3 sentences) so the model learns concise replies.
# ──────────────────────────────────────────────────────────────────────────────

QA: list[tuple[list[str], str]] = [

    # ── General / identity ───────────────────────────────────────────────────

    (["What are you?",
      "What is this device?",
      "Tell me about yourself.",
      "What is Hardware One?"],
     "I am Hardware One, an ESP32-based device. "
     "I can connect to sensors, display data, talk to other devices over ESP-NOW, "
     "and be controlled from a web browser or the command line."),

    (["What can you do?",
      "What are your capabilities?",
      "What features do you have?"],
     "I can connect to WiFi, run a web UI, talk to sensors, send data over MQTT, "
     "run automations, and pair with other Hardware One devices over ESP-NOW. "
     "Type help to see all available commands."),

    (["Who made you?",
      "Who built Hardware One?"],
     "Hardware One is an open-source ESP32 firmware project. "
     "It runs on ESP-IDF, not the Arduino IDE."),

    (["What boards do you run on?",
      "Which ESP32 boards are supported?",
      "What hardware can run this firmware?"],
     "Hardware One runs on the Seeed XIAO ESP32-S3, Adafruit QT PY ESP32-S3, "
     "Adafruit QT PY ESP32, and Adafruit Feather ESP32."),

    (["What deployment types are there?",
      "What configurations can Hardware One run in?"],
     "There are four configurations: Barebones headless node, Sensor Appliance, "
     "Standard Handheld with OLED and gamepad, and Bonded Microcontrollers "
     "where two devices share command registries."),

    # ── System commands ──────────────────────────────────────────────────────

    (["How do I reboot the device?",
      "How do I restart?",
      "What is the reboot command?"],
     "Type reboot to restart the device."),

    (["How do I check system status?",
      "How do I see the system status?"],
     "Type status to see the current system status including WiFi, sensors, and uptime."),

    (["How do I check uptime?",
      "How long has the device been running?"],
     "Type uptime to see how long the device has been running since the last reboot."),

    (["How do I check memory usage?",
      "How do I see RAM usage?",
      "How much memory is being used?"],
     "Type memory to see heap and PSRAM usage. "
     "Use memsum for a one-line summary or memreport for a full breakdown."),

    (["How do I see all available commands?",
      "How do I get help?",
      "What commands are available?"],
     "Type help to enter the interactive help system. "
     "Type a module name to see its commands, or help all to list everything."),

    (["How do I clear the CLI?",
      "How do I clear command history?"],
     "Type clear to clear the CLI history."),

    (["How do I broadcast a message to all users?",
      "How do I send a message to everyone?"],
     "Type broadcast followed by your message to send it to all connected users. "
     "This requires admin privileges."),

    # ── WiFi ─────────────────────────────────────────────────────────────────

    (["How do I connect to WiFi?",
      "How do I get online?",
      "How do I join a network?"],
     "First save your credentials with setssid and setpass, "
     "then type wifi connect to connect to the saved network."),

    (["How do I check WiFi status?",
      "Is the device connected to WiFi?",
      "What is the WiFi status?"],
     "Type wifi to show the current WiFi connection status, IP address, and signal strength."),

    (["How do I scan for WiFi networks?",
      "How do I see nearby WiFi networks?"],
     "Type wifi scan to scan for nearby access points and list them with their signal strength."),

    (["How do I set my WiFi password?",
      "How do I save WiFi credentials?"],
     "Type setssid followed by your network name to save the SSID, "
     "then setpass followed by your password to save it. "
     "Then type wifi connect to connect."),

    (["How do I turn WiFi off?",
      "How do I disable WiFi?"],
     "Type wifi off to disable WiFi. Use wifi on to re-enable it."),

    (["How do I disconnect from WiFi?",
      "How do I leave the current network?"],
     "Type wifi disconnect to disconnect from the current network."),

    (["How do I sync the clock?",
      "How do I update the time?",
      "How do I sync NTP?"],
     "Type synctime to synchronise the device clock from an NTP server. "
     "You must be connected to WiFi first. Type time to see the current time."),

    # ── Web UI ───────────────────────────────────────────────────────────────

    (["How do I access the web UI?",
      "How do I open the web interface?",
      "How do I use the browser interface?"],
     "Connect the device to WiFi, then navigate to its IP address in a browser. "
     "Type webstatus to see the IP address. Log in with your admin credentials."),

    (["How do I start the web server?",
      "How do I enable the web UI?"],
     "Type webstart to start the web server. "
     "Type webauto on to make it start automatically on every boot."),

    (["How do I stop the web server?",
      "How do I turn off the web UI?"],
     "Type webstop to stop the web server."),

    (["How do I make the web server start on boot?",
      "How do I auto-start the web server?"],
     "Type webauto on to enable auto-start. The web server will start automatically after every reboot."),

    (["What can I do in the web UI?",
      "What tabs are in the web interface?"],
     "The web UI has tabs for Sensors, ESP-NOW, Pair, MQTT, Settings, and a full CLI. "
     "You can view live sensor data, manage peers, configure MQTT, and run any command from the browser."),

    # ── OLED / display ───────────────────────────────────────────────────────

    (["How do I turn the OLED on?",
      "How do I turn the screen on?"],
     "Type oled on to turn the OLED display on."),

    (["How do I turn the OLED off?",
      "How do I turn the screen off?"],
     "Type oled off to turn the OLED display off."),

    (["How do I set OLED brightness?",
      "How do I change the screen brightness?"],
     "Type oled brightness followed by a value from 0 to 255 to set the brightness."),

    (["How do I open the OLED menu?",
      "How do I get to the main menu?"],
     "Type oled menu to open the main menu on the OLED display."),

    (["What is on the OLED main menu?",
      "What sections does the OLED menu have?"],
     "The OLED main menu has sections for Network, Sensors, System, Settings, Logging, and Power. "
     "Navigate it with the Seesaw gamepad joystick and buttons."),

    # ── LED ──────────────────────────────────────────────────────────────────

    (["How do I change the LED color?",
      "How do I set the NeoPixel color?",
      "How do I change the RGB LED?"],
     "Type ledcolor followed by a color name to set the onboard NeoPixel LED. "
     "For example, ledcolor red or ledcolor blue."),

    (["How do I turn off the LED?",
      "How do I turn off the NeoPixel?"],
     "Type ledcolor off to turn off the onboard NeoPixel LED."),

    # ── Sensors (general) ────────────────────────────────────────────────────

    (["How do I use a sensor?",
      "How do sensors work?",
      "What is the pattern for sensor commands?"],
     "All sensors follow the same pattern: openSENSOR to start it, "
     "closeSENSOR to stop it, and SENSORread to take a single reading. "
     "For example, opentof, closetof, and tofread for the time-of-flight sensor."),

    (["How do I auto-start a sensor on boot?",
      "How do I make a sensor start automatically?"],
     "Type SENSORautostart on to enable auto-start for that sensor. "
     "For example, imuautostart on to start the IMU automatically on every boot."),

    (["How do I scan I2C devices?",
      "How do I see what I2C devices are connected?"],
     "Type i2c scan to scan the I2C bus and list all detected devices with their addresses."),

    # ── Sensors (specific) ───────────────────────────────────────────────────

    (["How do I use the time-of-flight sensor?",
      "How does the ToF sensor work?",
      "How do I measure distance?"],
     "Type opentof to start the VL53L4CX time-of-flight sensor, then tofread to take a reading. "
     "It measures distance up to 6 metres and supports up to 4 simultaneous measurements."),

    (["How do I use the thermal camera?",
      "How do I get thermal images?",
      "How does the MLX90640 work?"],
     "Type openthermal to start the MLX90640 32 by 24 thermal camera. "
     "The web UI shows a live heatmap. Type closethermal to stop it."),

    (["How do I use the IMU?",
      "How do I get orientation data?",
      "How does the BNO055 work?"],
     "Type openimu to start the BNO055 9-DoF orientation sensor, then imuread to take a reading. "
     "It provides accelerometer, gyroscope, and magnetometer data fused into orientation angles."),

    (["How do I use the GPS?",
      "How do I get location data?",
      "How does GPS work on this device?"],
     "Type opengps to start the PA1010D GPS module, then gpsread to get the current position. "
     "The web UI includes an offline map viewer for viewing GPS tracks."),

    (["How do I use the gesture sensor?",
      "How do I detect gestures?",
      "How does the APDS9960 work?"],
     "The APDS9960 has three modes: apdscolor for RGB light readings, "
     "apdsproximity for proximity detection, and apdsgesture for up, down, left, and right gesture detection."),

    (["How do I use the FM radio?",
      "How do I tune the radio?",
      "How does the FM radio work?"],
     "Type openfmradio to start the RDA5807 FM radio, "
     "then fmradio tune followed by the frequency in MHz to tune to a station. "
     "For example, fmradio tune 101.5."),

    (["How do I use the presence sensor?",
      "How do I detect if someone is in the room?"],
     "Type openpresence to start the STHS34PF80 IR presence sensor, "
     "then presenceread to check for presence or motion in the area."),

    (["How do I use the RTC?",
      "How do I use the hardware clock?"],
     "Type openrtc to start the DS3231 real-time clock, then rtcread to get the current time from the hardware clock."),

    (["How do I use the servo controller?",
      "How do I control servos?"],
     "Use servo followed by the channel number and angle to move a servo. "
     "The PCA9685 controller supports up to 16 servo channels."),

    (["How do I log sensor data?",
      "How do I record sensor readings?"],
     "Type sensorlog start followed by the sensor name to begin logging to a file. "
     "Use sensorlog stop to stop, sensorlog view to see recent entries, and sensorlog list to see active logs."),

    # ── ESP-NOW ──────────────────────────────────────────────────────────────

    (["What is ESP-NOW?",
      "How does the mesh network work?",
      "What is the ESP-NOW feature?"],
     "ESP-NOW is Hardware One's wireless peer-to-peer protocol. "
     "Devices pair with a shared passphrase and form an encrypted mesh network. "
     "They can send messages, transfer files, and share metadata with each other."),

    (["How do I pair with another device?",
      "How do I connect two Hardware One devices?"],
     "Go to the Pair tab in the web UI on both devices, set the same passphrase on both, "
     "then have one device initiate pairing while the other accepts."),

    (["How do I check ESP-NOW status?",
      "How do I see my paired devices?"],
     "Type espnow status to see the ESP-NOW connection status and the list of paired peers."),

    (["How do I send a message to another device?",
      "How do I send text to a peer?"],
     "Type espnow send followed by the peer's MAC address and your message. "
     "You can find peer MAC addresses with espnow peers."),

    (["How do I list paired devices?",
      "How do I see all my peers?"],
     "Type espnow peers to list all known paired devices with their MAC addresses and names."),

    (["How do I set my device name?",
      "How do I name this device?"],
     "Type espnow setname followed by the name you want to give this device. "
     "The name is shared with peers during metadata sync."),

    (["How do I set the room for this device?",
      "How do I assign a room?"],
     "Type espnow setroom followed by the room name. "
     "Room information is used in automation conditions and metadata sync."),

    (["How do I set a zone?",
      "How do I assign a zone to the device?"],
     "Type espnow setzone followed by the zone name. "
     "Zones can be used in automation conditions, for example IF ZONE=Upstairs."),

    (["How do I transfer a file to another device?",
      "How do I send a file over ESP-NOW?"],
     "Type espnow sendfile followed by the peer MAC address and the file path to transfer a file to another device."),

    (["What is bonded mode?",
      "How does device bonding work?"],
     "In bonded mode, two devices share their command registries. "
     "The controller device gains a Remote tab in its web UI showing the paired device's features. "
     "Commands are routed automatically to whichever device can execute them."),

    # ── MQTT ─────────────────────────────────────────────────────────────────

    (["What is MQTT?",
      "How does MQTT work on this device?",
      "What is the MQTT feature?"],
     "MQTT lets the device publish sensor data and receive commands via a message broker "
     "like Home Assistant Mosquitto. Configure the broker address in the web UI or with the mqtt commands."),

    (["How do I connect to an MQTT broker?",
      "How do I set up MQTT?"],
     "Set the broker with mqtt broker followed by the host address, "
     "then mqtt connect to connect. "
     "Use mqtt user and mqtt pass if authentication is required."),

    (["How do I check MQTT connection status?",
      "Is MQTT connected?"],
     "Type mqtt status to see whether the device is connected to the MQTT broker "
     "and what topic prefix is being used."),

    (["How do I disconnect from MQTT?"],
     "Type mqtt disconnect to disconnect from the current MQTT broker."),

    (["How do I set the MQTT topic?",
      "How do I change the MQTT topic prefix?"],
     "Type mqtt topic followed by your desired prefix to set the MQTT topic prefix. "
     "All published messages will use this prefix."),

    # ── Automations ──────────────────────────────────────────────────────────

    (["What are automations?",
      "How do automations work?",
      "What is the automation feature?"],
     "Automations are scheduled or conditional command sequences that run locally on the device. "
     "They require no internet connection and can trigger at specific times, intervals, or on boot."),

    (["How do I list my automations?",
      "How do I see all automations?"],
     "Type automation list to see all configured automations and their current status."),

    (["How do I run an automation immediately?",
      "How do I trigger an automation manually?"],
     "Type automation run followed by the automation name to execute it immediately."),

    (["How do I enable or disable an automation?",
      "How do I turn an automation on or off?"],
     "Type automation enable followed by the name to enable it, "
     "or automation disable followed by the name to disable it."),

    (["How do I delete an automation?",
      "How do I remove an automation?"],
     "Type automation delete followed by the automation name to remove it."),

    (["How do I create an automation that runs at a specific time?",
      "How do I schedule a daily task?"],
     "Add an automation with SCHEDULE: TIME=HH:MM to run it daily at that time. "
     "For example, TIME=08:00 runs the automation every day at 8 in the morning."),

    (["How do I create an automation that runs every few minutes?",
      "How do I set a repeating interval?"],
     "Use SCHEDULE: INTERVAL=Xm in your automation where X is the number of minutes. "
     "You can also use Xs for seconds or Xh for hours."),

    (["How do I make something happen at boot?",
      "How do I run a command when the device starts?"],
     "Add an automation with SCHEDULE: BOOT to run it once every time the device boots."),

    (["How do I make an automation with a condition?",
      "How do conditional automations work?"],
     "Use IF followed by a condition and THEN followed by commands. "
     "For example, IF TEMP>75 THEN ledcolor red will turn the LED red when temperature exceeds 75."),

    # ── Filesystem ───────────────────────────────────────────────────────────

    (["How do I see what files are on the device?",
      "How do I list files?"],
     "Type files to list files in the root directory, or files followed by a path to list a specific folder."),

    (["How do I check storage space?",
      "How much storage is available?"],
     "Type fsusage to see the total and available space on the filesystem."),

    (["How do I view a file?",
      "How do I read a file on the device?"],
     "Type fileview followed by the file path to display its contents."),

    (["How do I delete a file?",
      "How do I remove a file?"],
     "Type filedelete followed by the file path to delete it."),

    (["How do I create a directory?",
      "How do I make a folder?"],
     "Type mkdir followed by the path to create a new directory."),

    # ── Users / auth ─────────────────────────────────────────────────────────

    (["How do I add a user?",
      "How do I create a new user account?"],
     "Type users add followed by the username and password to create a new user account. "
     "This requires admin privileges."),

    (["How do I list users?",
      "How do I see all user accounts?"],
     "Type users list to see all user accounts and their roles."),

    (["How do I change a password?",
      "How do I update a user password?"],
     "Type users passwd followed by the username and new password to change it."),

    (["How do I make someone an admin?",
      "How do I promote a user?"],
     "Type users promote followed by the username to grant them admin privileges."),

    (["How do I delete a user?",
      "How do I remove a user account?"],
     "Type users delete followed by the username to remove their account."),

    # ── Settings ─────────────────────────────────────────────────────────────

    (["How do I see all settings?",
      "How do I list device settings?"],
     "Type settings list to see all current device settings and their values."),

    (["How do I change a setting?",
      "How do I update a device setting?"],
     "Type settings set followed by the key and value to change a setting. "
     "For example, settings set deviceName MyDevice."),

    (["How do I reset settings to defaults?",
      "How do I factory reset the settings?"],
     "Type settings reset to reset all settings to their default values."),

    (["How do I export my settings?",
      "How do I back up settings?"],
     "Type settings export to export all current settings as JSON. "
     "You can save this output and use it to restore settings later."),

    # ── Debug ────────────────────────────────────────────────────────────────

    (["How do I enable debug output?",
      "How do I turn on debug logging?"],
     "Type debug followed by the module name and 1 to enable it. "
     "For example, debugwifi 1 enables WiFi debug logging. "
     "Add temp at the end to enable it only until the next reboot."),

    (["How do I disable debug output?",
      "How do I turn off debug logging?"],
     "Type debug followed by the module name and 0 to disable it. "
     "For example, debugwifi 0 turns off WiFi debug logging."),

    (["What debug modules are available?",
      "What can I debug?"],
     "Debug modules include debughttp, debugwifi, debugespnow, debugmqtt, debugautomations, "
     "debugsensors, debugstorage, debugcli, debugauth, and more. "
     "Type debug list to see all available flags."),

    # ── First time setup ─────────────────────────────────────────────────────

    (["How do I set up the device for the first time?",
      "What happens when I first boot Hardware One?",
      "How do I do first-time setup?"],
     "On first boot the device launches a setup wizard automatically on serial and OLED. "
     "Open serial monitor at 115200 baud. Choose Basic for quick setup or Advanced to configure everything. "
     "You will create an admin account with a username and password."),

    (["What is the default baud rate for serial?",
      "What baud rate do I use for the serial monitor?"],
     "Connect the serial monitor at 115200 baud."),

    (["How do I access the device from a browser for the first time?",
      "How do I find the device IP address?"],
     "After connecting to WiFi, the device prints its IP address in the serial monitor. "
     "You can also type webstatus at any time to see the IP address and web server status."),

    # ── Bluetooth ────────────────────────────────────────────────────────────

    (["How do I enable Bluetooth?",
      "How do I turn on BLE?"],
     "Type bluetooth on to enable BLE. "
     "Note that Bluetooth must be enabled in the build configuration with ENABLE_BLUETOOTH=1."),

    (["How do I check Bluetooth status?",
      "Is Bluetooth enabled?"],
     "Type bluetooth status to see whether BLE is enabled and any current connection status."),

    (["How do I connect to a BLE device?",
      "How do I pair with Bluetooth?"],
     "Type bluetooth scan to find nearby BLE devices, "
     "then bluetooth connect followed by the device address to connect."),

    # ── I2C ──────────────────────────────────────────────────────────────────

    (["How do I reset the I2C bus?",
      "The I2C sensor is not responding, what do I do?"],
     "Type i2c reset to reset the I2C bus. "
     "This can help if a sensor becomes unresponsive without needing to reboot."),

    (["How do I change I2C bus speed?",
      "How do I set I2C speed?"],
     "Type i2c speed followed by the speed in Hz to set the I2C bus speed."),

    # ── Boards and build ─────────────────────────────────────────────────────

    (["How do I switch between ESP32 and ESP32-S3?",
      "How do I change the target board?"],
     "Run idf.py fullclean first, then idf.py set-target esp32s3 or idf.py set-target esp32. "
     "Using the wrong target for your board can cause boot failures due to different PSRAM modes."),

    (["How do I build and flash the firmware?",
      "How do I flash Hardware One?"],
     "Run idf.py build to build, then idf.py -p PORT flash to flash. "
     "Replace PORT with your device's serial port. "
     "You can combine them with idf.py -p PORT flash monitor to also open the serial monitor."),

    (["How do I enable a feature in the build?",
      "How do I turn on a sensor in the firmware?"],
     "Edit System_BuildConfig.h in the components/hardwareone folder. "
     "Set the flag for the feature you want to 1, then rebuild and reflash."),

    # ── Battery ──────────────────────────────────────────────────────────────

    (["How do I check battery level?",
      "How much battery is left?",
      "What is the battery status?"],
     "Type battery status to see the current voltage, charge level, and charging status. "
     "Battery monitoring must be enabled in the build configuration."),
]

# ──────────────────────────────────────────────────────────────────────────────
# FORMAT TEMPLATES
# Each exchange is wrapped in one of these templates to add variety.
# ──────────────────────────────────────────────────────────────────────────────

TEMPLATES = [
    "Q: {q}\nA: {a}",
    "Question: {q}\nAnswer: {a}",
    "User: {q}\nHardwareOne: {a}",
    "{q}\n{a}",
]


def build_corpus(pairs: list[tuple[list[str], str]], repeat: int) -> str:
    lines: list[str] = []
    flat = [(q, a) for questions, a in pairs for q in questions]

    for _ in range(repeat):
        random.shuffle(flat)
        for q, a in flat:
            tmpl = random.choice(TEMPLATES)
            lines.append(tmpl.format(q=q, a=a))
            lines.append("")   # blank line between exchanges

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Hardware One Q&A fine-tuning corpus")
    ap.add_argument("--out",    default="hardwareone_qa.txt", help="Output file path")
    ap.add_argument("--repeat", type=int, default=4,
                    help="Times to cycle through all pairs (default 4)")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    corpus = build_corpus(QA, args.repeat)

    out = Path(args.out)
    out.write_text(corpus, encoding="utf-8")

    total_pairs  = sum(len(qs) for qs, _ in QA)
    total_lines  = corpus.count("\n")
    total_chars  = len(corpus)
    print(f"Topics:       {len(QA)}")
    print(f"Q&A pairs:    {total_pairs} unique  x{args.repeat} repeats = {total_pairs * args.repeat} total exchanges")
    print(f"Output lines: {total_lines:,}")
    print(f"Output size:  {total_chars / 1024:.1f} KB")
    print(f"Written to:   {out.resolve()}")
    print()
    print("Next step — fine-tune your pre-trained stretch model:")
    print(f"  python3 train_tiny_model_gpu.py \\")
    print(f"      --preset stretch \\")
    print(f"      --text {out} \\")
    print(f"      --finetune-from ./out_stretch \\")
    print(f"      --epochs 5 \\")
    print(f"      --lr 1e-4 \\")
    print(f"      --batch-size 32 \\")
    print(f"      --out ./out_stretch_finetuned")


if __name__ == "__main__":
    main()
