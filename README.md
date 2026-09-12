# PRIME - AI-Powered GNSS-Denied Navigation System

> AI/ML-based Intelligent Dead Reckoning for seamless vehicle navigation during GNSS-denied conditions.

## Overview

PRIME is an AI-powered navigation system designed to maintain continuous vehicle position estimation when GNSS signals become unavailable or unreliable.

The system combines IMU sensor data, GNSS information, and temporal deep learning to estimate motion and reduce the accumulated drift of conventional inertial dead reckoning.

## Problem

GNSS can become unavailable or unreliable in environments such as:

- Tunnels
- Urban canyons
- Indoor or covered environments
- Remote or obstructed areas
- GNSS interference or blackout conditions

During these periods, conventional IMU-based dead reckoning accumulates positioning errors over time.

## Proposed Solution

PRIME uses a Temporal CNN-based AI model to learn motion patterns from sequential sensor data.

### Core Pipeline

```text
IMU + GNSS Data
       |
       v
Data Preprocessing
       |
       v
Temporal Sequence Creation
       |
       v
Temporal CNN
       |
       v
Motion / Position Estimation
       |
       v
Drift Correction
       |
       v
Navigation Output