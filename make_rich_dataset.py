#!/usr/bin/env python3
"""
make_rich_dataset.py — generate a rich, multi-format Hardware One training corpus.

Goes beyond simple Q&A pairs to include:
  1. Q&A pairs (with EOS delimiters between each)
  2. Descriptive passages (teach grammar, fluency, sentence structure)
  3. Semantic clusters (word association — teach which words belong together)
  4. Sentence completions (teach natural sentence flow)
  5. Corrective pairs (teach what NOT to say)
  6. Command reference lines (dense factual recall)
  7. Conversational chains (multi-turn exchanges)

Usage:
    python3 make_rich_dataset.py [--out hardwareone_rich.txt] [--repeat 3]
"""

import argparse
import random
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1. Q&A PAIRS  (existing format, but answers are tighter and more varied)
# ──────────────────────────────────────────────────────────────────────────────

QA: list[tuple[list[str], str]] = [

    # ── Identity ──────────────────────────────────────────────────────────────

    (["What are you?",
      "What is this device?",
      "Tell me about yourself.",
      "What is Hardware One?"],
     "I am Hardware One, an ESP32-S3 device with WiFi, sensors, a web dashboard, "
     "and ESP-NOW mesh networking."),

    (["What can you do?",
      "What are your capabilities?",
      "What features do you have?",
      "List your features."],
     "I connect to WiFi, run a web UI, read sensors, send data over MQTT, "
     "run automations, and pair with other devices over ESP-NOW."),

    (["Who made you?",
      "Who built Hardware One?",
      "Who created this?"],
     "Hardware One is an open-source ESP32 firmware project built on ESP-IDF."),

    (["What boards do you run on?",
      "Which ESP32 boards are supported?",
      "What hardware can run this?"],
     "Hardware One runs on the Seeed XIAO ESP32-S3, Adafruit QT PY ESP32-S3, "
     "Adafruit QT PY ESP32, and Adafruit Feather ESP32."),

    (["What deployment types are there?",
      "What configurations can Hardware One run in?",
      "What modes does Hardware One have?"],
     "Four configurations: Barebones headless node, Sensor Appliance, "
     "Standard Handheld with OLED and gamepad, and Bonded Microcontrollers."),

    # ── WiFi ──────────────────────────────────────────────────────────────────

    (["How do I connect to WiFi?",
      "How do I join a network?",
      "How do I set up WiFi?"],
     "Type wifi connect followed by the network name and password. "
     "The device saves credentials and reconnects automatically on reboot."),

    (["What is the WiFi status?",
      "Is WiFi connected?",
      "How do I check WiFi?"],
     "Type wifi to show the current connection status, IP address, and signal strength."),

    (["How do I disconnect from WiFi?",
      "How do I leave the current network?"],
     "Type wifi disconnect to disconnect from the current network."),

    (["How do I scan for networks?",
      "How do I find WiFi networks?"],
     "Type wifi scan to see a list of nearby WiFi networks with signal strength."),

    (["How do I set up the access point?",
      "How do I use AP mode?",
      "How do I create a hotspot?"],
     "Type wifi ap to start the built-in access point. Connect to it from your phone "
     "or laptop to access the web dashboard without an existing network."),

    # ── Sensors ───────────────────────────────────────────────────────────────

    (["What sensors are supported?",
      "What sensors can I use?",
      "Which sensors work with Hardware One?"],
     "Supported sensors include BME280 for temperature and humidity, ICM-42688-P IMU, "
     "VL53L4CX time-of-flight, MLX90640 thermal camera, MLX90614 thermometer, "
     "APDS9960 gesture sensor, and GPS."),

    (["How do I read a sensor?",
      "How do I get sensor data?",
      "How do I check sensor readings?"],
     "Type the sensor name to see its current reading. For example, type bme to see "
     "temperature, humidity, and pressure from the BME280."),

    (["How do I start a sensor?",
      "How do I turn on a sensor?",
      "How do I open a sensor?"],
     "Type open followed by the sensor name. For example, openbme starts the BME280, "
     "openimu starts the IMU, and opentof starts the time-of-flight sensor."),

    (["How do I stop a sensor?",
      "How do I close a sensor?",
      "How do I turn off a sensor?"],
     "Type close followed by the sensor name. For example, closebme stops the BME280."),

    (["How do I auto-start a sensor on boot?",
      "How do I make a sensor start automatically?"],
     "Type SENSORautostart on. For example, imuautostart on starts the IMU "
     "automatically on every boot."),

    (["How do I get thermal images?",
      "How does the thermal camera work?"],
     "Type openthermal to start the MLX90640 thermal camera. The web UI shows "
     "a live 32 by 24 heatmap. Type closethermal to stop it."),

    (["How do I use the time-of-flight sensor?",
      "How does distance measurement work?"],
     "Type opentof to start the VL53L4CX sensor. It measures distance up to 6 metres."),

    (["How do I read the IMU?",
      "How do I check orientation?",
      "How does the motion sensor work?"],
     "Type openimu to start the ICM-42688-P, then imu to read acceleration and gyroscope data."),

    (["What is the temperature?",
      "How warm is it?",
      "What is the current temperature?"],
     "Type bme to read the temperature from the BME280 sensor. "
     "The reading includes temperature, humidity, and barometric pressure."),

    # ── ESP-NOW ───────────────────────────────────────────────────────────────

    (["How do I pair with another device?",
      "How does ESP-NOW pairing work?",
      "How do I set up mesh networking?"],
     "Go to the Pair tab in the web UI on both devices. Set the same passphrase, "
     "then have one device initiate pairing while the other accepts."),

    (["How do I send a message over ESP-NOW?",
      "How do I talk to another device?"],
     "Type espnow send followed by the peer MAC address and your message."),

    (["How do I see my paired devices?",
      "How do I check ESP-NOW status?",
      "Who am I paired with?"],
     "Type espnow status to see the connection status and the list of paired peers."),

    (["How do I send a file over ESP-NOW?",
      "Can I transfer files between devices?"],
     "Type espnow sendfile followed by the peer MAC address and the file path."),

    (["What is ESP-NOW?",
      "How does peer-to-peer communication work?"],
     "ESP-NOW is a wireless protocol that lets devices communicate directly "
     "without a WiFi router. It works up to about 200 metres in open air."),

    # ── MQTT ──────────────────────────────────────────────────────────────────

    (["How do I connect to MQTT?",
      "How do I set up MQTT?",
      "How do I use MQTT?"],
     "Set the broker with mqtt broker followed by the host address, "
     "then mqtt connect. Use mqtt user and mqtt pass if authentication is needed."),

    (["Is MQTT connected?",
      "What is the MQTT status?"],
     "Type mqtt status to see whether the device is connected to the broker "
     "and what topic prefix is being used."),

    (["How do I publish to MQTT?",
      "How do I send MQTT data?"],
     "Type mqtt publish followed by the topic and message. The device automatically "
     "publishes sensor data to configured topics."),

    (["How do I subscribe to MQTT topics?",
      "How do I receive MQTT messages?"],
     "Type mqtt subscribe followed by the topic pattern. Incoming messages appear "
     "in the serial log and can trigger automations."),

    # ── BLE ────────────────────────────────────────────────────────────────────

    (["How do I use Bluetooth?",
      "How do I pair with Bluetooth?",
      "How does BLE work?"],
     "Type bluetooth scan to find nearby BLE devices, then bluetooth connect "
     "followed by the device address to connect."),

    (["How do I scan for Bluetooth devices?",
      "How do I find nearby devices?"],
     "Type bluetooth scan to discover nearby BLE devices. Results show the device "
     "name, address, and signal strength."),

    # ── System ────────────────────────────────────────────────────────────────

    (["How do I reboot the device?",
      "How do I restart?",
      "What is the reboot command?"],
     "Type reboot to restart the device."),

    (["How do I check memory usage?",
      "How do I see RAM usage?",
      "How much memory is free?"],
     "Type memory to see heap and PSRAM usage. Use memsum for a one-line summary."),

    (["How do I check uptime?",
      "How long has the device been running?"],
     "Type uptime to see how long the device has been running since the last reboot."),

    (["How do I check system status?",
      "How do I see what is running?"],
     "Type status to see system status including WiFi, sensors, and uptime."),

    (["How do I see all commands?",
      "What commands are available?",
      "How do I get help?"],
     "Type help to see all available commands. Type help followed by a command name "
     "for detailed usage information."),

    (["How do I update the firmware?",
      "How does OTA work?",
      "How do I do a firmware update?"],
     "Go to the Settings tab in the web UI and use the OTA update section to upload "
     "a new firmware binary. The device reboots automatically after a successful update."),

    (["How do I check the firmware version?",
      "What version am I running?"],
     "Type version to see the current firmware version and build date."),

    # ── Web UI ────────────────────────────────────────────────────────────────

    (["How do I access the web UI?",
      "How do I open the dashboard?",
      "How do I use the web interface?"],
     "Connect to the same WiFi network as the device, then open its IP address "
     "in a web browser. The dashboard shows sensors, files, settings, and more."),

    (["What tabs are in the web UI?",
      "What does the dashboard show?"],
     "The web dashboard has tabs for Sensors, Files, Settings, Pair, MQTT, GPS, "
     "and LLM. Each tab shows real-time data and controls for that feature."),

    (["How do I log in to the web UI?",
      "What is the web password?"],
     "The default login uses the credentials set during first-time setup. "
     "You can change the password in the Settings tab."),

    # ── Files ─────────────────────────────────────────────────────────────────

    (["How do I browse files?",
      "How do I see what files are on the device?"],
     "Go to the Files tab in the web UI or type ls to list files in the current directory."),

    (["How do I upload a file?",
      "How do I transfer files to the device?"],
     "Use the Files tab in the web UI to drag and drop files, or use the upload button."),

    (["How do I delete a file?",
      "How do I remove a file?"],
     "Type rm followed by the file path, or use the delete button in the Files tab."),

    (["Does Hardware One support SD cards?",
      "How do I use an SD card?"],
     "Insert a micro SD card and the device mounts it automatically at /sd. "
     "The SD card is used for logs, map tiles, model files, and large data storage."),

    # ── Automations ───────────────────────────────────────────────────────────

    (["How do I create an automation?",
      "How do automations work?",
      "How do I set up a rule?"],
     "Use IF followed by a condition and THEN followed by commands. For example, "
     "IF TEMP>30 THEN ledcolor red turns the LED red when temperature exceeds 30."),

    (["What can trigger an automation?",
      "What conditions can automations use?"],
     "Automations can trigger on sensor values, time of day, MQTT messages, "
     "or button presses. Conditions can be combined with AND and OR."),

    (["How do I list automations?",
      "How do I see active rules?"],
     "Type automations to see all configured automation rules and their status."),

    # ── OLED Display ──────────────────────────────────────────────────────────

    (["How do I use the display?",
      "How does the OLED work?"],
     "The SSD1306 OLED shows status information, sensor readings, and menu options. "
     "Type oled menu to open the main menu."),

    (["How do I change screen brightness?",
      "How do I dim the display?"],
     "Type oled brightness followed by a value from 0 to 255."),

    # ── GPS ───────────────────────────────────────────────────────────────────

    (["How do I use GPS?",
      "How does GPS work?",
      "How do I get my location?"],
     "Type opengps to start the GPS module. Once it has a fix, type gps to see "
     "latitude, longitude, altitude, and speed. The web UI shows your position on a map."),

    (["How do I see the map?",
      "Where is the GPS map?"],
     "The GPS tab in the web UI shows your position on an interactive map. "
     "Map tiles are cached on the SD card for offline use."),

    # ── LLM ───────────────────────────────────────────────────────────────────

    (["How do I use the LLM?",
      "How does the language model work?",
      "How do I ask the AI a question?"],
     "Go to the LLM tab in the web UI and type your question. The model runs "
     "entirely on the device using PSRAM. Type llm ask followed by your question "
     "from the command line."),

    (["How do I load a model?",
      "How do I change the LLM model?"],
     "Place the model.bin file in /sd/llm/ on the SD card or /system/llm/ on "
     "internal storage. Type llm load to load it. Type llm models to see available models."),

    # ── Users ─────────────────────────────────────────────────────────────────

    (["How do I add a user?",
      "How do I create an account?"],
     "Type users add followed by the username, password, and role."),

    (["How do I list users?",
      "How do I see all accounts?"],
     "Type users list to see all user accounts and their roles."),

    (["How do I delete a user?",
      "How do I remove an account?"],
     "Type users delete followed by the username to remove their account."),

    # ── LED ───────────────────────────────────────────────────────────────────

    (["How do I change the LED color?",
      "How do I control the NeoPixel?"],
     "Type ledcolor followed by a color name or hex code. For example, "
     "ledcolor red or ledcolor FF00FF."),

    (["How do I turn off the LED?",
      "How do I disable the light?"],
     "Type ledcolor off or ledcolor 000000 to turn off the NeoPixel LED."),

    # ── Debug ─────────────────────────────────────────────────────────────────

    (["How do I enable debug logging?",
      "How do I see debug output?"],
     "Type debug followed by the module name and 1 to enable. For example, "
     "debugwifi 1 turns on WiFi debug logging."),

    (["How do I disable debug output?",
      "How do I turn off logging?"],
     "Type debug followed by the module name and 0 to disable."),

    # ── Settings ──────────────────────────────────────────────────────────────

    (["How do I change settings?",
      "Where are the settings?"],
     "Use the Settings tab in the web UI or type settings to see current values. "
     "Settings are saved in a JSON file on internal storage."),

    (["How do I export settings?",
      "How do I backup my configuration?"],
     "Use the Settings tab in the web UI to export settings as a JSON file. "
     "You can import them on another device for easy provisioning."),

    # ── Battery ───────────────────────────────────────────────────────────────

    (["How do I check battery level?",
      "How much battery is left?",
      "What is the battery status?"],
     "Type battery status to see voltage, charge level, and charging status."),

    # ── Building firmware ─────────────────────────────────────────────────────

    (["How do I build the firmware?",
      "How do I compile Hardware One?"],
     "Use idf.py build to compile the firmware. Flash with idf.py flash. "
     "Hardware One uses ESP-IDF, not the Arduino IDE."),

    (["How do I enable a feature in the build?",
      "How do I turn on a feature?"],
     "Edit System_BuildConfig.h and set the flag for the feature you want to 1, "
     "then rebuild and reflash."),

    # ── Bonded devices ────────────────────────────────────────────────────────

    (["What is bonded mode?",
      "How does bonding work?",
      "What are bonded microcontrollers?"],
     "Bonded mode connects two Hardware One devices so they share command registries. "
     "A command typed on one device can execute on the other transparently."),

    # ── I2C ───────────────────────────────────────────────────────────────────

    (["How do I scan for I2C devices?",
      "How do I find connected sensors?"],
     "Type i2cscan to scan the I2C bus. The device also scans automatically at startup "
     "and loads drivers for recognised sensors."),

    (["What is I2C?",
      "How does I2C work?"],
     "I2C is a two-wire bus that connects sensors and peripherals to the device. "
     "Multiple devices share the same two wires using different addresses."),

    # ── NTP / Time ────────────────────────────────────────────────────────────

    (["How do I set the time?",
      "How do I sync the clock?"],
     "Type synctime to synchronise from an NTP server. You need WiFi first. "
     "Type time to see the current time."),

    (["How do I use the hardware clock?",
      "How does the RTC work?"],
     "Type openrtc to start the DS3231 real-time clock, then rtcread to get the time. "
     "The RTC keeps time even when the device is powered off."),
]

# ──────────────────────────────────────────────────────────────────────────────
# 2. DESCRIPTIVE PASSAGES (teach grammar, fluency, multi-sentence coherence)
# ──────────────────────────────────────────────────────────────────────────────

PASSAGES: list[str] = [
    # WiFi
    "Hardware One connects to WiFi networks using the ESP32-S3 radio. "
    "When you type wifi connect followed by a network name and password, "
    "the device joins the network and gets an IP address. "
    "The credentials are saved so the device reconnects automatically after a reboot. "
    "You can check the connection by typing wifi, which shows the IP address and signal strength.",

    "The web dashboard is a browser-based interface served directly from the device. "
    "It does not need the internet or any cloud service. "
    "You access it by typing the device IP address into any web browser on the same network. "
    "The dashboard has tabs for sensors, files, settings, pairing, MQTT, GPS, and the language model.",

    # Sensors
    "Hardware One discovers I2C sensors automatically when it boots. "
    "It scans the I2C bus and loads drivers for any recognised device. "
    "Supported sensors include the BME280 for temperature and humidity, "
    "the ICM-42688-P IMU for motion and orientation, "
    "the VL53L4CX for distance measurement, "
    "and the MLX90640 thermal camera for infrared imaging.",

    "The BME280 sensor measures three things: temperature, humidity, and barometric pressure. "
    "Once started with openbme, you can read it by typing bme. "
    "The web UI shows live readings that update every few seconds. "
    "Sensor data can be published to MQTT or used in automation rules.",

    "The thermal camera is the MLX90640. It produces a 32 by 24 pixel heat map. "
    "Type openthermal to start it. The web dashboard shows the thermal image in real time. "
    "Type closethermal to stop the camera and free its memory.",

    "The time-of-flight sensor measures distance using an infrared laser. "
    "It can measure objects up to six metres away. "
    "Type opentof to start it and tofread to take a reading. "
    "The web dashboard shows a live distance graph.",

    # ESP-NOW
    "ESP-NOW is a wireless protocol from Espressif that allows direct device-to-device "
    "communication without a WiFi router. Two Hardware One devices can pair by setting "
    "the same passphrase in the web UI and initiating pairing. Once paired, they can "
    "send messages, transfer files, and share commands. The range is about 200 metres "
    "in open air.",

    "Bonded mode takes ESP-NOW pairing further. When two devices are bonded, "
    "they share their command registries. A command typed on one device can "
    "execute on the other transparently. This is useful for setups where one device "
    "is a sensor node and the other is a display unit.",

    # MQTT
    "MQTT is a lightweight messaging protocol used in IoT systems. "
    "Hardware One can connect to any MQTT broker and publish sensor data to topics. "
    "Set the broker address with mqtt broker, then type mqtt connect. "
    "Incoming messages can trigger automations. "
    "MQTT requires a WiFi connection to the network where the broker is running.",

    # Automations
    "The automation engine evaluates rules based on sensor values, time, or messages. "
    "A rule has the form IF condition THEN action. For example, "
    "IF TEMP>30 THEN ledcolor red turns the LED red when it gets hot. "
    "Conditions can be combined with AND and OR. "
    "Actions can publish MQTT messages, control LEDs, or run any CLI command.",

    # Storage
    "Hardware One uses two storage systems. Internal flash uses LittleFS for settings, "
    "web pages, and small files. An optional micro SD card provides larger storage "
    "for logs, map tiles, model files, and recorded data. "
    "The Files tab in the web UI lets you browse both storage areas.",

    # LLM
    "Hardware One can run a small language model entirely on the device. "
    "The model is loaded into PSRAM and uses no internet connection. "
    "Models are stored in LLM1 binary format on the SD card or internal storage. "
    "The inference engine runs the full transformer forward pass in C "
    "with no external library dependency.",

    # GPS
    "The GPS module provides latitude, longitude, altitude, and speed. "
    "Type opengps to start it and gps to read the current position. "
    "The web UI shows the position on an interactive map. "
    "Map tiles are cached on the SD card so the map works offline.",

    # System design
    "Hardware One is designed for reliability in field conditions. "
    "The firmware avoids dynamic heap allocation in hot paths and uses static buffers. "
    "PSRAM is reserved for large allocations like model weights and audio buffers. "
    "The partition table supports two application images for OTA rollback.",

    "Settings are stored as JSON on internal flash. They include WiFi credentials, "
    "MQTT configuration, device identity, sensor calibration, and feature flags. "
    "Settings can be exported and imported through the web UI. "
    "This makes it easy to provision multiple devices with the same configuration.",

    # OLED
    "The SSD1306 OLED display shows status information on the device itself. "
    "It supports a menu system navigated with the Seesaw gamepad. "
    "The display shows sensor readings, WiFi status, battery level, and system messages. "
    "Type oled menu to open the main menu, or oled brightness to adjust the screen.",

    # BLE
    "Bluetooth Low Energy scanning finds nearby BLE devices and displays their names, "
    "addresses, and signal strength. Type bluetooth scan to start a scan. "
    "BLE is used for proximity detection and connecting to BLE peripherals.",

    # Building
    "Hardware One is built with the ESP-IDF framework, not the Arduino IDE. "
    "To compile, run idf.py build. To flash, run idf.py flash. "
    "Features are enabled or disabled in System_BuildConfig.h. "
    "Each feature has a flag that can be set to 0 or 1.",

    # Debug
    "Debug logging helps diagnose issues. Each subsystem has its own debug flag. "
    "Type debug followed by the module name and 1 to enable, or 0 to disable. "
    "Debug output appears on the serial console. "
    "For persistent logs, the system can write to a file on the SD card.",
]

# ──────────────────────────────────────────────────────────────────────────────
# 3. SEMANTIC CLUSTERS (word association — teach which words go together)
# ──────────────────────────────────────────────────────────────────────────────

SEMANTIC_CLUSTERS: list[tuple[str, list[str]]] = [
    ("WiFi", ["network", "router", "connect", "SSID", "password", "signal", "IP address",
              "disconnect", "access point", "scan", "antenna", "2.4 GHz"]),
    ("sensors", ["I2C", "bus", "address", "read", "data", "temperature", "humidity",
                 "pressure", "motion", "distance", "thermal", "calibration"]),
    ("BME280", ["temperature", "humidity", "pressure", "environmental", "sensor",
                "degrees", "Celsius", "percent", "hectopascal"]),
    ("IMU", ["accelerometer", "gyroscope", "motion", "orientation", "tilt",
             "rotation", "axis", "ICM-42688-P", "degrees per second"]),
    ("ESP-NOW", ["peer", "pair", "mesh", "wireless", "direct", "MAC address",
                 "passphrase", "bonded", "send", "receive", "range"]),
    ("MQTT", ["broker", "publish", "subscribe", "topic", "message", "QoS",
              "connect", "IoT", "payload"]),
    ("GPS", ["latitude", "longitude", "altitude", "speed", "satellite", "fix",
             "NMEA", "map", "position", "coordinates"]),
    ("display", ["OLED", "SSD1306", "screen", "brightness", "menu", "pixel",
                 "text", "128x64"]),
    ("storage", ["flash", "SD card", "LittleFS", "file", "upload", "download",
                 "directory", "path", "mount"]),
    ("automation", ["rule", "condition", "trigger", "action", "IF", "THEN",
                    "sensor value", "threshold", "AND", "OR"]),
    ("LED", ["NeoPixel", "color", "red", "green", "blue", "brightness",
             "RGB", "strip", "off"]),
    ("Bluetooth", ["BLE", "scan", "device", "address", "signal strength",
                   "RSSI", "proximity", "peripheral"]),
    ("system", ["reboot", "uptime", "memory", "heap", "PSRAM", "status",
                "version", "firmware", "ESP32-S3"]),
    ("web", ["dashboard", "browser", "tab", "settings", "login", "password",
             "API", "REST", "HTTP"]),
    ("LLM", ["model", "inference", "PSRAM", "tokens", "generate", "transformer",
             "weights", "quantized", "prompt"]),
    ("building", ["ESP-IDF", "compile", "flash", "build", "firmware",
                  "partition", "configuration", "idf.py"]),
    ("security", ["login", "password", "user", "role", "admin", "authentication",
                  "credentials"]),
    ("communication", ["WiFi", "ESP-NOW", "Bluetooth", "MQTT", "serial",
                       "wireless", "protocol", "network"]),
    ("time", ["NTP", "clock", "RTC", "DS3231", "sync", "timezone", "uptime"]),
    ("thermal camera", ["MLX90640", "infrared", "heat map", "32x24", "temperature",
                        "imaging", "pixel"]),
]

# ──────────────────────────────────────────────────────────────────────────────
# 4. SENTENCE COMPLETIONS (teach natural flow)
# ──────────────────────────────────────────────────────────────────────────────

COMPLETIONS: list[tuple[str, str]] = [
    ("Hardware One connects to WiFi by", "typing wifi connect followed by the network name and password."),
    ("To see sensor readings, you can", "type the sensor name or open the Sensors tab in the web UI."),
    ("ESP-NOW lets two devices", "communicate directly without a WiFi router."),
    ("The web dashboard is accessed by", "opening the device IP address in a browser on the same network."),
    ("MQTT is used to", "publish sensor data to a broker and receive messages from other systems."),
    ("The OLED display shows", "status information, sensor readings, and a navigable menu."),
    ("Automations trigger when", "a sensor value crosses a threshold or a scheduled time arrives."),
    ("The LLM runs entirely on", "the device using PSRAM, with no internet connection required."),
    ("Settings are stored as", "a JSON file on internal flash and can be exported from the web UI."),
    ("Debug logging is enabled by", "typing debug followed by the module name and 1."),
    ("Map tiles are cached on", "the SD card so GPS maps work offline in the field."),
    ("The thermal camera produces", "a 32 by 24 pixel heat map displayed in the web UI."),
    ("Bonded mode allows two devices to", "share command registries and route commands between each other."),
    ("The firmware is built using", "ESP-IDF with idf.py build and flashed with idf.py flash."),
    ("Battery status shows the current", "voltage, charge level, and whether the device is charging."),
    ("The time-of-flight sensor measures", "distance to objects up to six metres away using an infrared laser."),
    ("I2C sensors are detected by", "scanning the bus at startup and loading drivers for recognised addresses."),
    ("Files can be uploaded through", "the Files tab in the web UI by dragging and dropping."),
    ("NTP synchronisation sets the clock by", "connecting to a time server over WiFi. Type synctime to start."),
    ("User accounts control", "who can access the web dashboard and what commands they can run."),
    ("The GPS module provides", "latitude, longitude, altitude, and speed from satellite signals."),
    ("To reboot the device, simply", "type reboot in the command line or serial console."),
    ("Memory usage can be checked by", "typing memory for heap and PSRAM usage, or memsum for a summary."),
    ("Hardware One is designed for", "field deployment where internet access may not be available."),
    ("The command line interface is", "accessible over serial or through the web terminal in the dashboard."),
]

# ──────────────────────────────────────────────────────────────────────────────
# 5. CORRECTIVE PAIRS (teach what NOT to say — misconceptions)
# ──────────────────────────────────────────────────────────────────────────────

CORRECTIONS: list[tuple[str, str]] = [
    ("Wrong: Hardware One needs the internet to work.\n"
     "Right: Hardware One works offline. It serves its own web dashboard and runs a local LLM.",
     "The device is fully self-contained. WiFi connects to a local network, not the internet."),

    ("Wrong: Hardware One uses the Arduino IDE.\n"
     "Right: Hardware One uses ESP-IDF as its build system and framework.",
     "ESP-IDF provides better control over memory, tasks, and partitioning than Arduino."),

    ("Wrong: ESP-NOW needs a WiFi router.\n"
     "Right: ESP-NOW is peer-to-peer and works without any router or access point.",
     "Devices communicate directly using MAC addresses."),

    ("Wrong: MQTT works without WiFi.\n"
     "Right: MQTT requires a WiFi connection to reach the broker.",
     "The broker runs on a server on the network. Without WiFi, the device cannot reach it."),

    ("Wrong: The LLM needs an internet connection.\n"
     "Right: The LLM runs entirely on the device using PSRAM.",
     "Model weights are stored locally and inference happens on the ESP32-S3."),

    ("Wrong: Sensors are connected over WiFi.\n"
     "Right: Sensors connect over the I2C bus using physical wires.",
     "I2C is a two-wire protocol. Sensors are detected by scanning bus addresses."),

    ("Wrong: The display needs a web browser.\n"
     "Right: The OLED display is a physical screen on the device itself.",
     "The SSD1306 shows information locally without any network connection."),

    ("Wrong: Settings are stored in the cloud.\n"
     "Right: Settings are stored as a JSON file on internal flash storage.",
     "All configuration is local. Settings can be exported and imported through the web UI."),

    ("Wrong: The thermal camera shows visible light.\n"
     "Right: The MLX90640 measures infrared radiation to create a heat map.",
     "It shows temperature differences, not visible light images."),

    ("Wrong: GPS needs WiFi to work.\n"
     "Right: GPS receives signals directly from satellites and works without any network.",
     "WiFi is only needed to sync the map tiles. Position tracking works fully offline."),
]

# ──────────────────────────────────────────────────────────────────────────────
# 6. COMMAND REFERENCE (dense factual lines for recall)
# ──────────────────────────────────────────────────────────────────────────────

COMMAND_REFS: list[str] = [
    "Command: wifi connect SSID PASSWORD — join a WiFi network.",
    "Command: wifi disconnect — leave the current WiFi network.",
    "Command: wifi scan — scan for nearby WiFi networks.",
    "Command: wifi ap — start the access point.",
    "Command: wifi — show WiFi status and IP address.",
    "Command: reboot — restart the device.",
    "Command: status — show system status.",
    "Command: uptime — show time since last reboot.",
    "Command: memory — show heap and PSRAM usage.",
    "Command: memsum — one-line memory summary.",
    "Command: help — list all commands.",
    "Command: version — show firmware version.",
    "Command: openbme — start the BME280 sensor.",
    "Command: closebme — stop the BME280 sensor.",
    "Command: bme — read temperature, humidity, and pressure.",
    "Command: openimu — start the IMU sensor.",
    "Command: imu — read acceleration and gyroscope data.",
    "Command: opentof — start the time-of-flight sensor.",
    "Command: tofread — take a distance reading.",
    "Command: openthermal — start the thermal camera.",
    "Command: closethermal — stop the thermal camera.",
    "Command: opengps — start the GPS module.",
    "Command: gps — read current position.",
    "Command: espnow status — show ESP-NOW peers.",
    "Command: espnow send MAC MESSAGE — send a message to a peer.",
    "Command: espnow sendfile MAC PATH — send a file to a peer.",
    "Command: mqtt broker HOST — set the MQTT broker address.",
    "Command: mqtt connect — connect to the MQTT broker.",
    "Command: mqtt status — show MQTT connection status.",
    "Command: mqtt publish TOPIC MESSAGE — publish an MQTT message.",
    "Command: mqtt subscribe TOPIC — subscribe to an MQTT topic.",
    "Command: bluetooth scan — scan for BLE devices.",
    "Command: bluetooth connect ADDRESS — connect to a BLE device.",
    "Command: ledcolor COLOR — set the NeoPixel LED color.",
    "Command: oled menu — open the OLED main menu.",
    "Command: oled brightness VALUE — set display brightness 0-255.",
    "Command: debug MODULE 1 — enable debug logging for a module.",
    "Command: debug MODULE 0 — disable debug logging for a module.",
    "Command: settings — show current settings.",
    "Command: users list — list all user accounts.",
    "Command: users add NAME PASS ROLE — create a user account.",
    "Command: users delete NAME — delete a user account.",
    "Command: battery status — show battery voltage and charge.",
    "Command: synctime — sync clock from NTP server.",
    "Command: time — show current time.",
    "Command: openrtc — start the DS3231 real-time clock.",
    "Command: rtcread — read time from hardware clock.",
    "Command: i2cscan — scan the I2C bus for devices.",
    "Command: ls — list files in current directory.",
    "Command: rm PATH — delete a file.",
    "Command: llm load — load the LLM model.",
    "Command: llm ask QUESTION — ask the LLM a question.",
    "Command: llm models — list available model files.",
    "Command: llm unload — unload the model and free memory.",
    "Command: automations — list active automation rules.",
]

# ──────────────────────────────────────────────────────────────────────────────
# 7. CONVERSATIONAL CHAINS (multi-turn — teach coherent follow-up)
# ──────────────────────────────────────────────────────────────────────────────

CONVERSATIONS: list[list[tuple[str, str]]] = [
    [("How do I connect to WiFi?",
      "Type wifi connect followed by the network name and password."),
     ("What if the password is wrong?",
      "The device will fail to connect and show an error. Double-check the password and try again."),
     ("How do I know when it connects?",
      "Type wifi to see the status. When connected, it shows the IP address and signal strength.")],

    [("I want to read the temperature.",
      "Type openbme to start the BME280 sensor, then type bme to read it."),
     ("Can I see it on the web UI?",
      "Yes, the Sensors tab shows live temperature, humidity, and pressure readings."),
     ("Can I send the data to my server?",
      "Yes, set up MQTT with mqtt broker and mqtt connect. Sensor data is published to MQTT topics automatically.")],

    [("How do I pair two devices?",
      "Open the Pair tab on both devices. Set the same passphrase and initiate pairing."),
     ("What can they do once paired?",
      "They can send messages, transfer files, and share commands over ESP-NOW."),
     ("What is bonded mode?",
      "Bonded mode lets paired devices share command registries so commands on one execute on the other.")],

    [("The device is not responding.",
      "Try typing reboot to restart it. Check that the serial connection is active."),
     ("How do I check if it crashed?",
      "Type uptime to see if it recently rebooted. Check debug logs on the SD card for crash information."),
     ("How do I enable debug logs?",
      "Type debug followed by the module name and 1. Debug output goes to the serial console.")],

    [("How do I update the firmware?",
      "Go to the Settings tab in the web UI and use the OTA section to upload a new firmware binary."),
     ("What if the update fails?",
      "The device keeps the old firmware and can roll back. Two firmware slots ensure recovery."),
     ("Do I need to reconfigure after updating?",
      "No. Settings are stored separately from the firmware and persist across updates.")],

    [("I want to use the LLM.",
      "Place a model.bin file on the SD card at /sd/llm/ and type llm load."),
     ("How big can the model be?",
      "The model must fit in PSRAM. With 8 MB of PSRAM, models up to about 6 MB work well."),
     ("How do I ask it something?",
      "Type llm ask followed by your question, or use the LLM tab in the web UI.")],

    [("How do I create an automation rule?",
      "Type IF followed by a condition and THEN followed by a command. For example, IF TEMP>30 THEN ledcolor red."),
     ("Can I use multiple conditions?",
      "Yes, combine conditions with AND or OR. For example, IF TEMP>30 AND HUMIDITY>80 THEN ledcolor red."),
     ("How do I remove a rule?",
      "Type automations to see the list, then delete the rule by its number.")],
]

# ──────────────────────────────────────────────────────────────────────────────
# FORMAT TEMPLATES (for Q&A pairs)
# ──────────────────────────────────────────────────────────────────────────────

QA_TEMPLATES = [
    "Q: {q}\nA: {a}",
    "Q: {q}\nA: {a}",   # weight Q:/A: format more heavily — it's the inference format
    "Q: {q}\nA: {a}",
    "Question: {q}\nAnswer: {a}",
]


# ──────────────────────────────────────────────────────────────────────────────
# BUILD THE CORPUS
# ──────────────────────────────────────────────────────────────────────────────

def build_corpus(repeat: int) -> str:
    blocks: list[str] = []

    for _ in range(repeat):
        # 1. Q&A pairs (with EOS-like blank line separation)
        flat_qa = [(q, a) for questions, a in QA for q in questions]
        random.shuffle(flat_qa)
        for q, a in flat_qa:
            tmpl = random.choice(QA_TEMPLATES)
            blocks.append(tmpl.format(q=q, a=a))

        # 2. Descriptive passages
        passages = list(PASSAGES)
        random.shuffle(passages)
        for p in passages:
            blocks.append(p)

        # 3. Semantic clusters (as prose: "WiFi relates to: network, router, ...")
        clusters = list(SEMANTIC_CLUSTERS)
        random.shuffle(clusters)
        for topic, words in clusters:
            # Vary the format
            fmt = random.choice([
                "{topic}: {words}.",
                "{topic} is related to {words}.",
                "Words associated with {topic}: {words}.",
            ])
            blocks.append(fmt.format(topic=topic, words=", ".join(words)))

        # 4. Sentence completions
        comps = list(COMPLETIONS)
        random.shuffle(comps)
        for start, end in comps:
            blocks.append(f"{start} {end}")

        # 5. Corrective pairs
        corrs = list(CORRECTIONS)
        random.shuffle(corrs)
        for correction, explanation in corrs:
            blocks.append(f"{correction}\n{explanation}")

        # 6. Command reference (batch as a block each repeat)
        refs = list(COMMAND_REFS)
        random.shuffle(refs)
        for ref in refs:
            blocks.append(ref)

        # 7. Conversational chains
        convos = list(CONVERSATIONS)
        random.shuffle(convos)
        for chain in convos:
            lines = []
            for q, a in chain:
                lines.append(f"Q: {q}\nA: {a}")
            blocks.append("\n".join(lines))

    # Shuffle all blocks together so the model sees varied data types interleaved
    random.shuffle(blocks)

    return "\n\n".join(blocks) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate rich Hardware One training corpus")
    ap.add_argument("--out", default="hardwareone_rich.txt", help="Output file path")
    ap.add_argument("--repeat", type=int, default=3,
                    help="Times to cycle through all data (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    corpus = build_corpus(args.repeat)

    out = Path(args.out)
    out.write_text(corpus, encoding="utf-8")

    n_qa = sum(len(qs) for qs, _ in QA)
    n_lines = corpus.count("\n")
    n_chars = len(corpus)

    print(f"Data types included:")
    print(f"  Q&A pairs:           {n_qa} unique ({n_qa * args.repeat} total)")
    print(f"  Passages:            {len(PASSAGES)} unique ({len(PASSAGES) * args.repeat} total)")
    print(f"  Semantic clusters:   {len(SEMANTIC_CLUSTERS)} unique ({len(SEMANTIC_CLUSTERS) * args.repeat} total)")
    print(f"  Completions:         {len(COMPLETIONS)} unique ({len(COMPLETIONS) * args.repeat} total)")
    print(f"  Corrections:         {len(CORRECTIONS)} unique ({len(CORRECTIONS) * args.repeat} total)")
    print(f"  Command refs:        {len(COMMAND_REFS)} unique ({len(COMMAND_REFS) * args.repeat} total)")
    print(f"  Conversations:       {len(CONVERSATIONS)} unique ({len(CONVERSATIONS) * args.repeat} total)")
    print(f"")
    print(f"Output lines: {n_lines:,}")
    print(f"Output size:  {n_chars / 1024:.1f} KB")
    print(f"Written to:   {out.resolve()}")
    print()
    print("Next step — train with this data:")
    print(f"  python3 train_tiny_model_gpu.py \\")
    print(f"      --preset narrow2 \\")
    print(f"      --text {out} hardwareone_overview.txt \\")
    print(f"      --epochs 8 --lr 3e-4 --batch-size 16 \\")
    print(f"      --out ./out_narrow2_rich")


if __name__ == "__main__":
    main()
