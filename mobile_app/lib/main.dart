import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:geolocator/geolocator.dart';

void main() {
  runApp(const PrimeNavigationApp());
}

class PrimeNavigationApp extends StatefulWidget {
  const PrimeNavigationApp({super.key});

  @override
  State<PrimeNavigationApp> createState() =>
      _PrimeNavigationAppState();
}

class _PrimeNavigationAppState
    extends State<PrimeNavigationApp> {

  // ============================================================
  // PRIME API
  // ============================================================

  // API endpoint is configurable at build time.
  // Example:
  // Example:
// flutter run --dart-define=PRIME_API_URL=http://YOUR_PC_IP:8001
  static const String apiBaseUrl = String.fromEnvironment(
    'PRIME_API_URL',
    defaultValue: 'http://localhost:8001',
  );

  static const String apiUrl =
      '$apiBaseUrl/navigate';

  // ============================================================
  // SENSOR STREAMS
  // ============================================================

  StreamSubscription? accelSub;
  StreamSubscription? userAccelSub;
  StreamSubscription? gyroSub;
  StreamSubscription? magnetSub;
  StreamSubscription? locationSub;

  // ============================================================
  // SENSOR VALUES
  // ============================================================

  List<double> accel = [0, 0, 0];
  List<double> userAccel = [0, 0, 0];
  List<double> gyro = [0, 0, 0];
  List<double> magnet = [0, 0, 0];
  List<double> gravity = [0, 0, 9.80665];

  Position? gpsPosition;

  // ============================================================
  // GPS / PRIME ANCHOR
  // ============================================================

  bool hasGpsFix = false;

  double latitude = 0.0;
  double longitude = 0.0;

  double anchorLatitude = 0.0;
  double anchorLongitude = 0.0;

  // ============================================================
  // PRIME BUFFER
  // ============================================================

  final List<List<double>> rawBuffer = [];

  Timer? sampleTimer;

  bool running = false;
  bool apiBusy = false;

  int step = 0;
  int _sampleCount = 0;

  double liveSpeed = 0.0;
  Timer? gnssMonitorTimer;
  int selectedPage = 0;

  // ============================================================
  // NAVIGATION STATE
  // ============================================================

  String mode = 'WAITING';
  String gnssState = 'UNKNOWN';

  bool gnssAvailable = true;

  double northDisplacement = 0.0;
  double eastDisplacement = 0.0;

  String apiStatus = 'Waiting for GPS...';

  // ============================================================
  // LIVE MAP
  // ============================================================

  final MapController mapController = MapController();

  final List<LatLng> positionTrail = [];

  double deviceHeading = 0.0;

  // ============================================================
  // INIT
  // ============================================================

  @override
  void initState() {
    super.initState();
    _startSensors();
    _startGnssMonitor();
    _startAutomaticSampling();
  }

  // ============================================================
  // SENSOR + GPS INITIALIZATION
  // ============================================================

  Future<void> _startSensors() async {

    // ------------------------------------------------------------
    // ACCELEROMETER
    // ------------------------------------------------------------

    accelSub = accelerometerEventStream().listen((event) {
      accel = [
        event.x,
        event.y,
        event.z,
      ];

      _updateGravity();
    });

    // ------------------------------------------------------------
    // USER ACCELEROMETER
    // ------------------------------------------------------------

    userAccelSub =
        userAccelerometerEventStream().listen((event) {
      userAccel = [
        event.x,
        event.y,
        event.z,
      ];

      _updateGravity();
    });

    // ------------------------------------------------------------
    // GYROSCOPE
    // ------------------------------------------------------------

    gyroSub = gyroscopeEventStream().listen((event) {
      gyro = [
        event.x,
        event.y,
        event.z,
      ];
    });

    // ------------------------------------------------------------
    // MAGNETOMETER / HEADING
    // ------------------------------------------------------------

    magnetSub =
        magnetometerEventStream().listen((event) {

      magnet = [
        event.x,
        event.y,
        event.z,
      ];

      // Compass heading.
      //
      // The -90 degree correction compensates for the
      // sensor-axis orientation used by this application.
      var heading =
          atan2(event.y, event.x) * 180 / pi;

      heading = (heading - 90.0) % 360.0;

      if (heading < 0) {
        heading += 360;
      }

      if (mounted) {
        setState(() {
          deviceHeading = heading;
        });
      }
    });

    // ============================================================
    // LOCATION SERVICE
    // ============================================================

    final serviceEnabled =
        await Geolocator.isLocationServiceEnabled();

    if (!serviceEnabled) {
      if (mounted) {
        setState(() {
          apiStatus =
              'Turn ON Location to get live GPS position';
        });
      }

      return;
    }

    // ============================================================
    // LOCATION PERMISSION
    // ============================================================

    var permission =
        await Geolocator.checkPermission();

    if (permission == LocationPermission.denied) {
      permission =
          await Geolocator.requestPermission();
    }

    if (permission == LocationPermission.denied ||
        permission ==
            LocationPermission.deniedForever) {

      if (mounted) {
        setState(() {
          apiStatus =
              'Location permission denied';
        });
      }

      return;
    }

    // ============================================================
    // LIVE GPS STREAM
    // ============================================================

    locationSub =
        Geolocator.getPositionStream(
      locationSettings:
          const LocationSettings(
        accuracy: LocationAccuracy.best,
        distanceFilter: 0,
      ),
    ).listen((p) {

      gpsPosition = p;
      hasGpsFix = true;
      liveSpeed = max(0.0, p.speed);

      // ----------------------------------------------------------
      // IMPORTANT:
      // While GNSS is available, use the REAL phone GPS position.
      // During PRIME_DR, preserve the PRIME estimated position.
      // ----------------------------------------------------------

      if (gnssAvailable) {

        latitude = p.latitude;
        longitude = p.longitude;

        // Keep the live trail moving with GPS.
        _addTrailPoint(
          LatLng(latitude, longitude),
        );

        if (mounted) {
          setState(() {
            apiStatus =
                'LIVE GPS • Accuracy '
                '${p.accuracy.toStringAsFixed(1)}m';
          });
        }

        // Follow real GPS on the map.
        _moveMapToCurrentPosition();
      }
    });

    // ============================================================
    // GET FIRST GPS FIX
    // ============================================================

    try {

      final firstPosition =
          await Geolocator.getCurrentPosition(
        locationSettings:
            const LocationSettings(
          accuracy: LocationAccuracy.best,
        ),
      );

      gpsPosition = firstPosition;
      hasGpsFix = true;
      liveSpeed = max(0.0, firstPosition.speed);

      // ----------------------------------------------------------
      // REAL STARTING POINT
      // ----------------------------------------------------------

      latitude = firstPosition.latitude;
      longitude = firstPosition.longitude;

      anchorLatitude = latitude;
      anchorLongitude = longitude;

      positionTrail.clear();

      positionTrail.add(
        LatLng(
          latitude,
          longitude,
        ),
      );

      // Center map on actual phone location.
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;

        if (latitude != 0.0 &&
            longitude != 0.0) {

          mapController.move(
            LatLng(
              latitude,
              longitude,
            ),
            16.5,
          );
        }
      });

      if (mounted) {
        setState(() {
          mode = 'GNSS';
          gnssState = 'GOOD';

          apiStatus =
              'GPS LOCKED • '
              '${latitude.toStringAsFixed(7)}, '
              '${longitude.toStringAsFixed(7)}';
        });
      }

    } catch (e) {

      if (mounted) {
        setState(() {
          apiStatus =
              'Waiting for GPS fix...';
        });
      }
    }
  }

  // ============================================================
  // AUTOMATIC PRIME SENSOR SAMPLING
  // ============================================================

  void _startAutomaticSampling() {
    if (sampleTimer?.isActive == true) return;

    running = true;
    mode = 'SENSOR MONITOR';

    sampleTimer = Timer.periodic(
      const Duration(milliseconds: 100),
      (_) => _captureSensorSample(),
    );
  }

  // ============================================================
  // AUTOMATIC GNSS / LOCATION STATUS MONITOR
  // ============================================================

  void _startGnssMonitor() {
    gnssMonitorTimer?.cancel();

    // Check the actual phone Location Service periodically.
    gnssMonitorTimer = Timer.periodic(
      const Duration(seconds: 1),
      (_) async {
        final serviceEnabled =
            await Geolocator.isLocationServiceEnabled();

        if (!mounted) return;

        if (!serviceEnabled) {
          if (gnssAvailable || gnssState != 'OUTAGE') {
            setState(() {
              gnssAvailable = false;
              gnssState = 'OUTAGE';
              mode = running ? 'PRIME_DR' : 'WAITING';
              apiStatus = running
                  ? 'LOCATION OFF — PRIME DEAD RECKONING ACTIVE'
                  : 'LOCATION OFF — waiting for GNSS';
            });
          }
        } else if (!gnssAvailable) {
          // Location service came back. Refresh the real GPS fix.
          try {
            final p = await Geolocator.getCurrentPosition(
              locationSettings: const LocationSettings(
                accuracy: LocationAccuracy.best,
              ),
            );

            gpsPosition = p;
            hasGpsFix = true;
            liveSpeed = max(0.0, p.speed);

            setState(() {
              gnssAvailable = true;
              gnssState = 'GOOD';
              mode = running ? 'GNSS' : 'WAITING';
              latitude = p.latitude;
              longitude = p.longitude;
              apiStatus = 'GNSS RESTORED • LIVE LOCATION';
            });

            _addTrailPoint(LatLng(latitude, longitude));
            _moveMapToCurrentPosition();
          } catch (_) {
            // Service is on, but a fix may not be available yet.
          }
        }
      },
    );
  }

  // ============================================================
  // MOVE MAP
  // ============================================================

  void _moveMapToCurrentPosition() {

    if (latitude == 0.0 ||
        longitude == 0.0) {
      return;
    }

    WidgetsBinding.instance.addPostFrameCallback((_) {

      if (!mounted) return;

      mapController.move(
        LatLng(
          latitude,
          longitude,
        ),
        16.5,
      );
    });
  }

  // ============================================================
  // ADD TRAIL POINT
  // ============================================================

  void _addTrailPoint(LatLng point) {

    if (point.latitude == 0.0 &&
        point.longitude == 0.0) {
      return;
    }

    if (positionTrail.isEmpty) {

      positionTrail.add(point);
      return;
    }

    final distance =
        const Distance().as(
      LengthUnit.Meter,
      positionTrail.last,
      point,
    );

    if (distance > 0.5) {
      positionTrail.add(point);
    }
  }

  // ============================================================
  // GRAVITY
  // ============================================================

  void _updateGravity() {

    gravity[0] = 0.95 * gravity[0] + 0.05 * accel[0];
    gravity[1] = 0.95 * gravity[1] + 0.05 * accel[1];
    gravity[2] = 0.95 * gravity[2] + 0.05 * accel[2];
  }

  // ============================================================
  // START PRIME
  // ============================================================

  void _startPrime() {
    // PRIME now starts automatically with the app.
    // Kept as a compatibility method; no UI calls it.
    if (!running) {
      _startAutomaticSampling();
    }
  }

  // ============================================================
  // CAPTURE SENSOR SAMPLE
  // ============================================================

  void _captureSensorSample() {

    // Sensors are always collected while the app is open.
    // API/CNN inference is independent of visible map position.

    final sample = <double>[

      // 0-2 Accelerometer
      accel[0],
      accel[1],
      accel[2],

      // 3-5 Gyroscope
      gyro[0],
      gyro[1],
      gyro[2],

      // 6-8 Gravity
      gravity[0],
      gravity[1],
      gravity[2],

      // 9-11 Magnetometer
      magnet[0],
      magnet[1],
      magnet[2],
    ];

    rawBuffer.add(sample);
    _sampleCount++;

    debugPrint(
      'PRIME SENSOR | '
      'ACC=${accel.map((v) => v.toStringAsFixed(2)).join(',')} | '
      'USER=${userAccel.map((v) => v.toStringAsFixed(2)).join(',')} | '
      'GYRO=${gyro.map((v) => v.toStringAsFixed(2)).join(',')} | '
      'MAG=${magnet.map((v) => v.toStringAsFixed(2)).join(',')} | '
      'BUFFER=${rawBuffer.length}/60',
    );

    if (rawBuffer.length > 60) {
      rawBuffer.removeAt(0);
    }

    // Send 60-sample window to PRIME once every second (10 samples @ 10Hz)
    if (rawBuffer.length >= 60 &&
        !apiBusy &&
        _sampleCount % 10 == 0) {

      _sendToPrime();
    }

    if (mounted) {
      setState(() {});
    }
  }

  // ============================================================
  // BUILD V7 FEATURES
  // ============================================================

  List<List<double>> _buildV7Sequence() {
    final result = <List<double>>[];

    final int windowSize = rawBuffer.length;
    final List<double> accN = List.filled(windowSize, 0.0);
    final List<double> accE = List.filled(windowSize, 0.0);
    final List<double> accV = List.filled(windowSize, 0.0);
    final List<double> dts = List.filled(windowSize, 0.1);

    final List<double> velN = List.filled(windowSize, 0.0);
    final List<double> velE = List.filled(windowSize, 0.0);
    final List<double> dispN = List.filled(windowSize, 0.0);
    final List<double> dispE = List.filled(windowSize, 0.0);

    // Pass 1: Compute 3D Earth-frame rotated accelerations
    for (int i = 0; i < windowSize; i++) {
      final s = rawBuffer[i];
      final ax = s[0], ay = s[1], az = s[2];
      final gravX = s[6], gravY = s[7], gravZ = s[8];
      final mx = s[9], my = s[10], mz = s[11];

      final linearX = ax - gravX;
      final linearY = ay - gravY;
      final linearZ = az - gravZ;

      final gNorm = sqrt(gravX * gravX + gravY * gravY + gravZ * gravZ);
      final gHatX = (gNorm < 1e-6) ? 0.0 : gravX / gNorm;
      final gHatY = (gNorm < 1e-6) ? 0.0 : gravY / gNorm;
      final gHatZ = (gNorm < 1e-6) ? 1.0 : gravZ / gNorm;

      // east_axis = cross(magnet, g_hat)
      double eX = my * gHatZ - mz * gHatY;
      double eY = mz * gHatX - mx * gHatZ;
      double eZ = mx * gHatY - my * gHatX;
      final eNorm = sqrt(eX * eX + eY * eY + eZ * eZ);
      if (eNorm > 1e-6) {
        eX /= eNorm;
        eY /= eNorm;
        eZ /= eNorm;
      } else {
        eX = 1.0;
        eY = 0.0;
        eZ = 0.0;
      }

      // north_axis = cross(g_hat, east_axis)
      double nX = gHatY * eZ - gHatZ * eY;
      double nY = gHatZ * eX - gHatX * eZ;
      double nZ = gHatX * eY - gHatY * eX;
      final nNorm = sqrt(nX * nX + nY * nY + nZ * nZ);
      if (nNorm > 1e-6) {
        nX /= nNorm;
        nY /= nNorm;
        nZ /= nNorm;
      } else {
        nX = 0.0;
        nY = 1.0;
        nZ = 0.0;
      }

      accN[i] = linearX * nX + linearY * nY + linearZ * nZ;
      accE[i] = linearX * eX + linearY * eY + linearZ * eZ;
      accV[i] = linearX * gHatX + linearY * gHatY + linearZ * gHatZ;
      dts[i] = 0.1;
    }

    // Pass 2: Window-relative physics integration starting at 0.0
    for (int i = 1; i < windowSize; i++) {
      velN[i] = velN[i - 1] + accN[i] * dts[i];
      velE[i] = velE[i - 1] + accE[i] * dts[i];
      dispN[i] = dispN[i - 1] + velN[i] * dts[i];
      dispE[i] = dispE[i - 1] + velE[i] * dts[i];
    }

    // Pass 3: Build 25 V7 Features array per sample
    for (int i = 0; i < windowSize; i++) {
      final s = rawBuffer[i];
      final ax = s[0], ay = s[1], az = s[2];
      final gx = s[3], gy = s[4], gz = s[5];
      final gravX = s[6], gravY = s[7], gravZ = s[8];
      final mx = s[9], my = s[10], mz = s[11];

      final linearX = ax - gravX;
      final linearY = ay - gravY;
      final linearZ = az - gravZ;

      final accMag = sqrt(linearX * linearX + linearY * linearY + linearZ * linearZ);
      final gyroMag = sqrt(gx * gx + gy * gy + gz * gz);
      final magMag = sqrt(mx * mx + my * my + mz * mz);
      final gravityMag = sqrt(gravX * gravX + gravY * gravY + gravZ * gravZ);
      final speedPhys = sqrt(velN[i] * velN[i] + velE[i] * velE[i]);

      result.add([
        ax, ay, az,                       // 0-2
        gx, gy, gz,                       // 3-5
        gravX, gravY, gravZ,             // 6-8
        mx, my, mz,                       // 9-11
        accN[i], accE[i], accV[i],        // 12-14
        accMag, gyroMag, magMag,          // 15-17
        velN[i], velE[i],                 // 18-19
        dispN[i], dispE[i],               // 20-21
        speedPhys,                        // 22
        dts[i],                           // 23
        gravityMag,                       // 24
      ]);
    }

    return result;
  }

  // ============================================================
  // SEND 60 × 25 TO PRIME
  // ============================================================

  Future<void> _sendToPrime() async {

    if (apiBusy ||
        rawBuffer.length < 60) {
      return;
    }

    apiBusy = true;

    if (mounted) {
      setState(() {
        apiStatus =
            'Sending 60 × 25 to PRIME...';
      });
    }

    try {

      final sequence =
          _buildV7Sequence();

      final last = sequence.last;

      debugPrint(
        'PRIME V7 INPUT | '
        'LIN=${last[12].toStringAsFixed(3)},'
        '${last[13].toStringAsFixed(3)},'
        '${last[14].toStringAsFixed(3)} | '
        'PHYS=${last[19].toStringAsFixed(3)},'
        '${last[20].toStringAsFixed(3)} | '
        'SPEED=${last[21].toStringAsFixed(3)}',
      );

      final gps =
          gpsPosition;

      final body =
          jsonEncode({

        'sensor_data': {
          'sequence': sequence,
        },

        // ------------------------------------------------------
        // GNSS STATE
        // ------------------------------------------------------

        'satellites':
            gnssAvailable ? 8 : 0,

        'accuracy':
            gnssAvailable
                ? (gps?.accuracy ?? 5.0)
                : 999.0,

        'time_since_update':
            gnssAvailable
                ? 0.0
                : 10.0,

        'step_dt': 1.0,

        // ------------------------------------------------------
        // REAL GPS COORDINATE
        // ------------------------------------------------------

        'gnss_latitude':
            gnssAvailable
                ? gps?.latitude
                : null,

        'gnss_longitude':
            gnssAvailable
                ? gps?.longitude
                : null,
      });

      final client =
          HttpClient();

      try {

        final request =
            await client.postUrl(
          Uri.parse(apiUrl),
        );

        request.headers.contentType =
            ContentType.json;

        request.write(body);

        final response =
            await request.close();

        final responseText =
            await response
                .transform(
                  utf8.decoder,
                )
                .join();

        if (response.statusCode == 200) {

          final data =
              jsonDecode(responseText);

          final movement = data['movement'];
          final rawNorth =
              (movement?['north_displacement_m'] ?? 0.0).toDouble();
          final rawEast =
              (movement?['east_displacement_m'] ?? 0.0).toDouble();

          final lastSequence = sequence.last;
          final speedPhys = lastSequence[22];

          debugPrint(
            'PRIME V7 OUTPUT | SPEED=${speedPhys.toStringAsFixed(3)}m/s | '
            'RAW_N=${rawNorth.toStringAsFixed(3)}m | '
            'RAW_E=${rawEast.toStringAsFixed(3)}m',
          );

          _processPrimeResponse(data);

        } else {

          if (mounted) {

            setState(() {

              apiStatus =
                  'API error '
                  '${response.statusCode}';
            });
          }
        }

      } finally {

        client.close();
      }

    } catch (e) {

      if (mounted) {

        setState(() {

          apiStatus =
              'API FAILED: $e';
        });
      }

    } finally {

      apiBusy = false;
    }
  }

  // ============================================================
  // PROCESS PRIME RESPONSE
  // ============================================================

  void _processPrimeResponse(
      Map<String, dynamic> data) {

    final navigation =
        data['navigation'];

    final movement =
        data['movement'];

    if (navigation == null) {
      return;
    }

    final newMode =
        navigation['mode']
                ?.toString() ??
            'UNKNOWN';

    final newGnss =
        navigation['gnss_state']
                ?.toString() ??
            'UNKNOWN';

    final rawNorth =
        (movement?[
                'north_displacement_m']
            as num?)
        ?.toDouble() ??
        0.0;

    final rawEast =
        (movement?[
                'east_displacement_m']
            as num?)
        ?.toDouble() ??
        0.0;

    final newStep =
        (data['step'] as num?)
            ?.toInt() ??
        step;

    // ----------------------------------------------------------
    // DURING GNSS:
    // REAL GPS IS THE POSITION SOURCE.
    //
    // DURING OUTAGE:
    // PRIME API POSITION IS THE POSITION SOURCE.
    // ----------------------------------------------------------

    // ----------------------------------------------------------
    // VISIBLE MAP POSITION
    //
    // The map is GPS/last-known-position only.
    // PRIME API/dead-reckoning output MUST NOT move the map.
    // ----------------------------------------------------------

    final double newLat =
        gpsPosition?.latitude ?? latitude;

    final double newLon =
        gpsPosition?.longitude ?? longitude;

    // ----------------------------------------------------------
    // FILTER LARGE SINGLE-STEP JUMPS
    // ----------------------------------------------------------

    double newNorth = northDisplacement + rawNorth;
    double newEast = eastDisplacement + rawEast;

    if (!mounted) {
      return;
    }

    final newPoint =
        LatLng(
          newLat,
          newLon,
        );

    _addTrailPoint(newPoint);

    setState(() {

      mode = newMode;

      gnssState =
          newGnss;

      latitude =
          newLat;

      longitude =
          newLon;

      northDisplacement =
          newNorth;

      eastDisplacement =
          newEast;

      if (!gnssAvailable) {
        liveSpeed = sqrt(
          rawNorth * rawNorth + rawEast * rawEast,
        );
      }

      step =
          newStep;

      apiStatus =
          'PRIME V7 • Step $step • '
          'dN: ${rawNorth.toStringAsFixed(2)}m • '
          'dE: ${rawEast.toStringAsFixed(2)}m';
    });

    // Do NOT move the visible map from PRIME/DR output.
    // The map follows real GNSS only; when GNSS is lost it freezes
    // at the last known location.
  }

  // ============================================================
  // GNSS SIMULATION
  // ============================================================

  void _toggleGnss() {
    // GNSS status is now automatic. The method is retained only
    // for compatibility and is intentionally not exposed in the UI.
  }

  // ============================================================
  // STOP PRIME
  // ============================================================

  void _stopPrime() {
    // Kept for compatibility. The UI no longer exposes STOP PRIME.
    // Sensor collection is intended to remain live.
  }

  // ============================================================
  // DISPOSE
  // ============================================================

  @override
  void dispose() {

    sampleTimer?.cancel();

    accelSub?.cancel();
    userAccelSub?.cancel();
    gyroSub?.cancel();
    magnetSub?.cancel();
    locationSub?.cancel();
    gnssMonitorTimer?.cancel();

    super.dispose();
  }

  // ============================================================
  // UI — PAGE 1 + PAGE 2
  // ============================================================

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF070B12),
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.cyan,
          brightness: Brightness.dark,
        ),
      ),
      home: Scaffold(
        backgroundColor: const Color(0xFF070B12),
        appBar: AppBar(
          elevation: 0,
          backgroundColor: const Color(0xFF0B111B),
          titleSpacing: 18,
          title: Row(
            children: [
              const Icon(
                Icons.navigation_rounded,
                color: Colors.cyanAccent,
                size: 25,
              ),
              const SizedBox(width: 9),
              const Text(
                'PRIME',
                style: TextStyle(
                  fontWeight: FontWeight.w800,
                  letterSpacing: 1.4,
                ),
              ),
              const Spacer(),
              _statusPill(
                gnssAvailable ? Icons.gps_fixed : Icons.gps_off,
                gnssAvailable ? 'GNSS ON' : 'GNSS OFF',
                gnssAvailable
                    ? Colors.greenAccent
                    : Colors.redAccent,
              ),
            ],
          ),
        ),
        body: IndexedStack(
          index: selectedPage,
          children: [
            _buildHomePage(),
            _buildSensorsPage(),
          ],
        ),
        bottomNavigationBar: NavigationBar(
          selectedIndex: selectedPage,
          onDestinationSelected: (index) {
            setState(() => selectedPage = index);
          },
          backgroundColor: const Color(0xFF0B111B),
          indicatorColor: Colors.cyan.withValues(alpha: 0.18),
          destinations: const [
            NavigationDestination(
              icon: Icon(Icons.map_outlined),
              selectedIcon: Icon(Icons.map),
              label: 'Navigation',
            ),
            NavigationDestination(
              icon: Icon(Icons.sensors_outlined),
              selectedIcon: Icon(Icons.sensors),
              label: 'Sensors',
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHomePage() {
    final hasPosition = latitude != 0.0 || longitude != 0.0;
    final sampleProgress =
        (rawBuffer.length / 60.0).clamp(0.0, 1.0);

    return SafeArea(
      child: LayoutBuilder(
        builder: (context, constraints) {
          final mapHeight = constraints.maxHeight * 0.48;

          return SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(14, 12, 14, 22),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // ------------------------------------------------------
                // 1. MAP — roughly 50% of the page
                // ------------------------------------------------------
                SizedBox(
                  height: mapHeight.clamp(280.0, 520.0),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(22),
                    child: Stack(
                      children: [
                        FlutterMap(
                          mapController: mapController,
                          options: MapOptions(
                            initialCenter: hasPosition
                                ? LatLng(latitude, longitude)
                                : const LatLng(28.5355, 77.3910),
                            initialZoom: 16.5,
                            interactionOptions:
                                const InteractionOptions(
                              flags: InteractiveFlag.all,
                            ),
                          ),
                          children: [
                            TileLayer(
                              urlTemplate:
                                  'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                              subdomains: const ['a', 'b', 'c'],
                              userAgentPackageName:
                                  'com.example.mobile_app',
                            ),
                            if (positionTrail.length >= 2)
                              PolylineLayer(
                                polylines: [
                                  Polyline(
                                    points: positionTrail,
                                    strokeWidth: 5,
                                    color: Colors.cyanAccent,
                                  ),
                                ],
                              ),
                            if (hasPosition)
                              MarkerLayer(
                                markers: [
                                  Marker(
                                    point:
                                        LatLng(latitude, longitude),
                                    width: 72,
                                    height: 72,
                                    child: Transform.rotate(
                                      angle:
                                          deviceHeading * pi / 180,
                                      child: Container(
                                        decoration: BoxDecoration(
                                          shape: BoxShape.circle,
                                          color: Colors.cyanAccent
                                              .withValues(alpha: 0.18),
                                          border: Border.all(
                                            color: Colors.cyanAccent,
                                            width: 2,
                                          ),
                                        ),
                                        child: const Icon(
                                          Icons.navigation_rounded,
                                          color: Colors.cyanAccent,
                                          size: 38,
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                          ],
                        ),

                        // GNSS automatic status
                        Positioned(
                          top: 12,
                          left: 12,
                          child: _statusPill(
                            gnssAvailable
                                ? Icons.gps_fixed
                                : Icons.gps_off,
                            gnssAvailable
                                ? 'GNSS ON'
                                : 'GNSS OFF',
                            gnssAvailable
                                ? Colors.greenAccent
                                : Colors.redAccent,
                            dark: true,
                          ),
                        ),

                        // Sensor navigation direction
                        Positioned(
                          top: 12,
                          right: 12,
                          child: _glassPill(
                            Icons.navigation_rounded,
                            '${deviceHeading.toStringAsFixed(0)}°',
                            'DIRECTION',
                          ),
                        ),

                        // When GNSS is off, explicitly explain the arrow
                        if (!gnssAvailable)
                          Positioned(
                            left: 12,
                            right: 12,
                            bottom: 12,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 13,
                                vertical: 10,
                              ),
                              decoration: BoxDecoration(
                                color: const Color(0xE6101722),
                                borderRadius:
                                    BorderRadius.circular(13),
                                border: Border.all(
                                  color:
                                      Colors.orangeAccent.withValues(
                                    alpha: 0.55,
                                  ),
                                ),
                              ),
                              child: const Row(
                                children: [
                                  Icon(
                                    Icons.sensors,
                                    color: Colors.orangeAccent,
                                    size: 18,
                                  ),
                                  SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      'GNSS OFF • direction maintained by PRIME sensors',
                                      style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),

                        Positioned(
                          bottom: 8,
                          right: 8,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 6,
                              vertical: 3,
                            ),
                            color: Colors.white.withValues(
                              alpha: 0.82,
                            ),
                            child: const Text(
                              '© OpenStreetMap contributors',
                              style: TextStyle(
                                color: Colors.black87,
                                fontSize: 8,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 12),

                // ------------------------------------------------------
                // 2. CURRENT LOCATION
                // ------------------------------------------------------
                _metricCard(
                  icon: Icons.location_on_rounded,
                  label: 'CURRENT LOCATION',
                  value: hasPosition
                      ? '${latitude.toStringAsFixed(6)}°\n${longitude.toStringAsFixed(6)}°'
                      : 'Waiting for fix',
                  footer: gnssAvailable
                      ? 'Live GNSS position'
                      : 'Last known position • map frozen',
                ),
                const SizedBox(height: 10),

                // ------------------------------------------------------
                // 4. COLLECTING SAMPLE + CNN RUNNING
                // ------------------------------------------------------
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: _cardDecoration(),
                  child: Column(
                    crossAxisAlignment:
                        CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          _liveDot(
                            running
                                ? Colors.greenAccent
                                : Colors.white38,
                          ),
                          const SizedBox(width: 10),
                          const Expanded(
                            child: Text(
                              'COLLECTING SAMPLES • AUTO',
                              style: TextStyle(
                                fontWeight: FontWeight.w800,
                                letterSpacing: 0.8,
                              ),
                            ),
                          ),
                          Text(
                            '${rawBuffer.length}/60',
                            style: const TextStyle(
                              color: Colors.cyanAccent,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 11),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(6),
                        child: LinearProgressIndicator(
                          value: sampleProgress,
                          minHeight: 7,
                          backgroundColor:
                              Colors.white.withValues(alpha: 0.08),
                        ),
                      ),
                      const SizedBox(height: 15),
                      Row(
                        children: [
                          const Icon(
                            Icons.memory_rounded,
                            color: Colors.cyanAccent,
                            size: 22,
                          ),
                          const SizedBox(width: 9),
                          const Expanded(
                            child: Column(
                              crossAxisAlignment:
                                  CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'PRIME TEMPORAL CNN',
                                  style: TextStyle(
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                                SizedBox(height: 2),
                                Text(
                                  '12 sensors • 60 timestep window',
                                  style: TextStyle(
                                    color: Colors.white54,
                                    fontSize: 12,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          _smallState(
                            apiBusy
                                ? 'INFERENCE'
                                : running
                                    ? 'RUNNING'
                                    : 'IDLE',
                            apiBusy || running
                                ? Colors.greenAccent
                                : Colors.white38,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 10),

                Container(
                  padding: const EdgeInsets.all(15),
                  decoration: _cardDecoration(),
                  child: Row(
                    children: [
                      _liveDot(
                        running
                            ? Colors.greenAccent
                            : Colors.white38,
                      ),
                      const SizedBox(width: 10),
                      const Expanded(
                        child: Column(
                          crossAxisAlignment:
                              CrossAxisAlignment.start,
                          children: [
                            Text(
                              'PRIME LIVE PROCESSING',
                              style: TextStyle(
                                fontWeight: FontWeight.w800,
                                letterSpacing: 0.5,
                              ),
                            ),
                            SizedBox(height: 3),
                            Text(
                              'Sensors collect automatically • CNN runs in background',
                              style: TextStyle(
                                color: Colors.white38,
                                fontSize: 10,
                              ),
                            ),
                          ],
                        ),
                      ),
                      _smallState(
                        'LIVE',
                        Colors.greenAccent,
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 8),

                Text(
                  apiStatus,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white38,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildSensorsPage() {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(14, 16, 14, 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Column(
                    crossAxisAlignment:
                        CrossAxisAlignment.start,
                    children: [
                      Text(
                        'LIVE SENSOR DATA',
                        style: TextStyle(
                          fontSize: 21,
                          fontWeight: FontWeight.w900,
                          letterSpacing: 0.8,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        '12 inputs • updating in real time',
                        style: TextStyle(
                          color: Colors.white54,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
                _smallState(
                  '12 ACTIVE',
                  Colors.greenAccent,
                ),
              ],
            ),

            const SizedBox(height: 16),

            _sensorSection(
              title: 'ACCELEROMETER',
              subtitle: 'Linear + gravitational acceleration source',
              icon: Icons.speed_rounded,
              values: [
                _sensorValue('AX', accel[0], 'm/s²'),
                _sensorValue('AY', accel[1], 'm/s²'),
                _sensorValue('AZ', accel[2], 'm/s²'),
              ],
            ),

            const SizedBox(height: 10),

            _sensorSection(
              title: 'GYROSCOPE',
              subtitle: 'Angular velocity',
              icon: Icons.rotate_right_rounded,
              values: [
                _sensorValue('GX', gyro[0], 'rad/s'),
                _sensorValue('GY', gyro[1], 'rad/s'),
                _sensorValue('GZ', gyro[2], 'rad/s'),
              ],
            ),

            const SizedBox(height: 10),

            _sensorSection(
              title: 'GRAVITY',
              subtitle: 'Gravity vector used for frame transformation',
              icon: Icons.south_rounded,
              values: [
                _sensorValue('GX', gravity[0], 'm/s²'),
                _sensorValue('GY', gravity[1], 'm/s²'),
                _sensorValue('GZ', gravity[2], 'm/s²'),
              ],
            ),

            const SizedBox(height: 10),

            _sensorSection(
              title: 'MAGNETOMETER',
              subtitle: 'Magnetic field + heading',
              icon: Icons.explore_rounded,
              values: [
                _sensorValue('MX', magnet[0], 'µT'),
                _sensorValue('MY', magnet[1], 'µT'),
                _sensorValue('MZ', magnet[2], 'µT'),
              ],
            ),

            const SizedBox(height: 14),

            Container(
              padding: const EdgeInsets.all(16),
              decoration: _cardDecoration(),
              child: Row(
                children: [
                  Container(
                    width: 46,
                    height: 46,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.cyan.withValues(alpha: 0.12),
                    ),
                    child: const Icon(
                      Icons.memory_rounded,
                      color: Colors.cyanAccent,
                    ),
                  ),
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment:
                          CrossAxisAlignment.start,
                      children: [
                        Text(
                          'PRIME MODEL INPUT',
                          style: TextStyle(
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        SizedBox(height: 4),
                        Text(
                          '12 raw sensor channels • 60 timesteps',
                          style: TextStyle(
                            color: Colors.white54,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Text(
                    '60 × 12',
                    style: const TextStyle(
                      color: Colors.cyanAccent,
                      fontSize: 16,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _sensorSection({
    required String title,
    required String subtitle,
    required IconData icon,
    required List<Widget> values,
  }) {
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 14, 14, 12),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(11),
                  color: Colors.cyan.withValues(alpha: 0.11),
                ),
                child: Icon(
                  icon,
                  color: Colors.cyanAccent,
                  size: 21,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        letterSpacing: 0.6,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: const TextStyle(
                        color: Colors.white38,
                        fontSize: 10,
                      ),
                    ),
                  ],
                ),
              ),
              const Icon(
                Icons.circle,
                color: Colors.greenAccent,
                size: 9,
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              for (int i = 0; i < values.length; i++) ...[
                Expanded(child: values[i]),
                if (i != values.length - 1)
                  const SizedBox(width: 8),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _sensorValue(
    String name,
    double value,
    String unit,
  ) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 10,
        vertical: 11,
      ),
      decoration: BoxDecoration(
        color: const Color(0xFF0A111B),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.07),
        ),
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Text(
            name,
            style: const TextStyle(
              color: Colors.white54,
              fontSize: 11,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            value.toStringAsFixed(3),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 17,
              fontWeight: FontWeight.w900,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            unit,
            style: const TextStyle(
              color: Colors.cyanAccent,
              fontSize: 9,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _metricCard({
    required IconData icon,
    required String label,
    required String value,
    required String footer,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                icon,
                color: Colors.cyanAccent,
                size: 19,
              ),
              const SizedBox(width: 7),
              Expanded(
                child: Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white54,
                    fontSize: 10,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            value,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w900,
            ),
          ),
          const SizedBox(height: 5),
          Text(
            footer,
            style: const TextStyle(
              color: Colors.white38,
              fontSize: 9,
            ),
          ),
        ],
      ),
    );
  }

  Widget _statusPill(
    IconData icon,
    String text,
    Color color, {
    bool dark = false,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 10,
        vertical: 7,
      ),
      decoration: BoxDecoration(
        color: dark
            ? const Color(0xDD08101A)
            : color.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: color.withValues(alpha: 0.55),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 15),
          const SizedBox(width: 6),
          Text(
            text,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.w900,
            ),
          ),
        ],
      ),
    );
  }

  Widget _glassPill(
    IconData icon,
    String value,
    String label,
  ) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 10,
        vertical: 8,
      ),
      decoration: BoxDecoration(
        color: const Color(0xDD08101A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.12),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: Colors.cyanAccent,
            size: 16,
          ),
          const SizedBox(width: 6),
          Column(
            crossAxisAlignment:
                CrossAxisAlignment.end,
            children: [
              Text(
                value,
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w900,
                ),
              ),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white38,
                  fontSize: 7,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _smallState(
    String text,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 9,
        vertical: 6,
      ),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(9),
        border: Border.all(
          color: color.withValues(alpha: 0.35),
        ),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 9,
          fontWeight: FontWeight.w900,
        ),
      ),
    );
  }

  Widget _liveDot(Color color) {
    return Container(
      width: 10,
      height: 10,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color,
        boxShadow: [
          BoxShadow(
            color: color.withValues(alpha: 0.55),
            blurRadius: 8,
          ),
        ],
      ),
    );
  }

  BoxDecoration _cardDecoration() {
    return BoxDecoration(
      color: const Color(0xFF0D151F),
      borderRadius: BorderRadius.circular(17),
      border: Border.all(
        color: Colors.white.withValues(alpha: 0.07),
      ),
    );
  }
}
