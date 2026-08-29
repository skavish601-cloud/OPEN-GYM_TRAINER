// Pose estimation service using MediaPipe
import '@mediapipe/pose';
const Pose = window.Pose;

export class PoseEstimator {
  constructor(onResults, options = {}) {
    this.onResults = onResults;
    this.options = {
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
      },
      selfieMode: true,
      upperBodyOnly: false,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
      ...options
    };

    this.pose = new Pose({
      locateFile: this.options.locateFile,
    });
    this.pose.setOptions(this.options);
    this.pose.onResults(this.onResults.bind(this));

    this.videoElement = null;
    this.isInitialized = false;
  }

  async initialize(videoElement) {
    this.videoElement = videoElement;
    this.isInitialized = true;
    return new Promise((resolve) => {
      this.videoElement.addEventListener('loadeddata', () => {
        resolve();
      });
    });
  }

  send(frame) {
    if (!this.isInitialized || !this.videoElement) return;
    this.pose.send({image: frame});
  }

  reset() {
    if (this.pose) {
      this.pose.reset();
    }
  }

  close() {
    if (this.pose) {
      this.pose.close();
      this.isInitialized = false;
    }
  }

  // Normalize landmarks to a consistent format
  static normalizeLandmarks(landmarks) {
    if (!landmarks) return {};

    const normalized = {};
    // Map MediaPipe pose landmarks to friendly names
    const landmarkMap = {
      'nose': 0,
      'left_eye_inner': 1,
      'left_eye': 2,
      'left_eye_outer': 3,
      'right_eye_inner': 4,
      'right_eye': 5,
      'right_eye_outer': 6,
      'left_ear': 7,
      'right_ear': 8,
      'mouth_left': 9,
      'mouth_right': 10,
      'left_shoulder': 11,
      'right_shoulder': 12,
      'left_elbow': 13,
      'right_elbow': 14,
      'left_wrist': 15,
      'right_wrist': 16,
      'left_pinky': 17,
      'right_pinky': 18,
      'left_index': 19,
      'right_index': 20,
      'left_thumb': 21,
      'right_thumb': 22,
      'left_hip': 23,
      'right_hip': 24,
      'left_knee': 25,
      'right_knee': 26,
      'left_ankle': 27,
      'right_ankle': 28,
      'left_heel': 29,
      'right_heel': 30,
      'left_foot_index': 31,
      'right_foot_index': 32
    };

    for (const [name, index] of Object.entries(landmarkMap)) {
      const landmark = landmarks[index];
      if (landmark) {
        normalized[name] = {
          x: landmark.x,
          y: landmark.y,
          z: landmark.z || 0,
          visibility: landmark.visibility || 0
        };
      }
    }

    return normalized;
  }

  // Calculate distance between two landmarks
  static calculateDistance(landmark1, landmark2) {
    if (!landmark1 || !landmark2) return 0;
    return Math.sqrt(
      Math.pow(landmark2.x - landmark1.x, 2) +
      Math.pow(landmark2.y - landmark1.y, 2) +
      Math.pow(landmark2.z - landmark1.z, 2)
    );
  }

  // Calculate angle between three landmarks (at landmark2)
  static calculateAngle(landmark1, landmark2, landmark3) {
    if (!landmark1 || !landmark2 || !landmark3) return 0;

    const a = { x: landmark1.x - landmark2.x, y: landmark1.y - landmark2.y };
    const b = { x: landmark3.x - landmark2.x, y: landmark3.y - landmark2.y };

    const dotProduct = a.x * b.x + a.y * b.y;
    const magnitudeA = Math.sqrt(a.x * a.x + a.y * a.y);
    const magnitudeB = Math.sqrt(b.x * b.x + b.y * b.y);

    if (magnitudeA === 0 || magnitudeB === 0) return 0;

    let angle = Math.acos(dotProduct / (magnitudeA * magnitudeB));
    return angle * (180 / Math.PI); // Convert to degrees
  }

  // Analyze squat form
  static analyzeSquat(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    const score = 100;

    // Check knee valgus (knees caving in)
    if (landmarks.left_knee && landmarks.right_knee &&
        landmarks.left_ankle && landmarks.right_ankle) {
      const kneeWidth = this.calculateDistance(landmarks.left_knee, landmarks.right_knee);
      const ankleWidth = this.calculateDistance(landmarks.left_ankle, landmarks.right_ankle);

      // If knees are significantly narrower than ankles, it's valgus
      if (kneeWidth < ankleWidth * 0.8) {
        issues.push('Knees caving in (valgus)');
      }
    }

    // Check depth (hip below knee)
    if (landmarks.left_hip && landmarks.right_hip &&
        landmarks.left_knee && landmarks.right_knee) {
      const hipY = (landmarks.left_hip.y + landmarks.right_hip.y) / 2;
      const kneeY = (landmarks.left_knee.y + landmarks.right_knee.y) / 2;

      // In selfie mode, y increases downward
      if (hipY > kneeY - 0.05) { // Hip not sufficiently below knee
        issues.push('Not deep enough');
      }
    }

    // Check forward lean (torso angle)
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_hip && landmarks.right_hip) {
      const shoulderCenter = {
        x: (landmarks.left_shoulder.x + landmarks.right_shoulder.x) / 2,
        y: (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2
      };
      const hipCenter = {
        x: (landmarks.left_hip.x + landmarks.right_hip.x) / 2,
        y: (landmarks.left_hip.y + landmarks.right_hip.y) / 2
      };

      const torsoAngle = Math.atan2(
        shoulderCenter.y - hipCenter.y,
        shoulderCenter.x - hipCenter.x
      ) * (180 / Math.PI);

      // Torso should be relatively vertical (around 0-20 degrees from vertical)
      if (Math.abs(torsoAngle) > 25) {
        issues.push('Excessive forward lean');
      }
    }

    return {
      issues,
      score: Math.max(0, score - issues.length * 20), // Simple scoring
      landmarks
    };
  }

  // Analyze deadlift form
  static analyzeDeadlift(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    const score = 100;

    // Check for rounded back
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_hip && landmarks.right_hip) {
      const shoulderCenter = {
        x: (landmarks.left_shoulder.x + landmarks.right_shoulder.x) / 2,
        y: (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2
      };
      const hipCenter = {
        x: (landmarks.left_hip.x + landmarks.right_hip.x) / 2,
        y: (landmarks.left_hip.y + landmarks.right_hip.y) / 2
      };

      // Calculate back angle
      const backAngle = Math.atan2(
        shoulderCenter.y - hipCenter.y,
        shoulderCenter.x - hipCenter.x
      ) * (180 / Math.PI);

      // Back should be relatively flat (not excessively curved)
      if (backAngle > 30) { // Too much upper back round
        issues.push('Rounded back');
      }
    }

    // Check hip position (should not squat the deadlift)
    if (landmarks.left_hip && landmarks.right_hip &&
        landmarks.left_knee && landmarks.right_knee) {
      const hipY = (landmarks.left_hip.y + landmarks.right_hip.y) / 2;
      const kneeY = (landmarks.left_knee.y + landmarks.right_knee.y) / 2;

      // In deadlift, hips should be higher than knees at start
      if (hipY > kneeY + 0.05) {
        issues.push('Hips too low (squatting the deadlift)');
      }
    }

    return {
      issues,
      score: Math.max(0, score - issues.length * 25),
      landmarks
    };
  }

  // Analyze bench press form
  static analyzeBenchPress(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    const score = 100;

    // Check for excessive shoulder elevation (shrugging)
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_ear && landmarks.right_ear) {
      const leftShoulderElevation = landmarks.left_shoulder.y - landmarks.left_ear.y;
      const rightShoulderElevation = landmarks.right_shoulder.y - landmarks.right_ear.y;
      const avgShoulderElevation = (leftShoulderElevation + rightShoulderElevation) / 2;

      // Shoulders should not be excessively elevated towards ears
      if (avgShoulderElevation < -0.05) { // Negative because y increases downward in selfie mode
        issues.push('Shoulders too elevated (shrugging)');
      }
    }

    // Check for uneven bar path (left/right symmetry)
    if (landmarks.left_wrist && landmarks.right_wrist &&
        landmarks.left_shoulder && landmarks.right_shoulder) {
      const wristWidth = this.calculateDistance(landmarks.left_wrist, landmarks.right_wrist);
      const shoulderWidth = this.calculateDistance(landmarks.left_shoulder, landmarks.right_shoulder);

      // Wrists should be roughly shoulder-width apart
      if (wristWidth < shoulderWidth * 0.8 || wristWidth > shoulderWidth * 1.2) {
        issues.push('Uneven grip width');
      }
    }

    // Check for excessive arch (hybrid of good form and potential injury risk)
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_hip && landmarks.right_hip) {
      const shoulderCenterY = (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2;
      const hipCenterY = (landmarks.left_hip.y + landmarks.right_hip.y) / 2;

      // Some arch is acceptable, but excessive arch can be problematic
      if (shoulderCenterY > hipCenterY + 0.1) { // Shoulders significantly below hips
        issues.push('Excessive back arch');
      }
    }

    return {
      issues,
      score: Math.max(0, score - issues.length * 15),
      landmarks
    };
  }

  // Analyze overhead press form
  static analyzeOverheadPress(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    const score = 100;

    // Check for excessive lower back arch
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_hip && landmarks.right_hip) {
      const shoulderCenterY = (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2;
      const hipCenterY = (landmarks.left_hip.y + landmarks.right_hip.y) / 2;

      // Excessive lean back
      if (shoulderCenterY > hipCenterY + 0.08) {
        issues.push('Excessive lean back');
      }
    }

    // Check for head position (should not jut forward)
    if (landmarks.nose && landmarks.left_ear && landmarks.right_ear) {
      const earCenterY = (landmarks.left_ear.y + landmarks.right_ear.y) / 2;
      const noseY = landmarks.nose.y;

      // Head should be relatively neutral, not jutting forward
      if (noseY < earCenterY - 0.05) { // Nose significantly above ears (jutting forward)
        issues.push('Head jutting forward');
      }
    }

    // Check for uneven elbow positioning
    if (landmarks.left_elbow && landmarks.right_elbow &&
        landmarks.left_shoulder && landmarks.right_shoulder) {
      const leftElbowHeight = landmarks.left_elbow.y - landmarks.left_shoulder.y;
      const rightElbowHeight = landmarks.right_elbow.y - landmarks.right_shoulder.y;
      const heightDiff = Math.abs(leftElbowHeight - rightElbowHeight);

      // Elbows should be at similar height
      if (heightDiff > 0.05) {
        issues.push('Uneven elbow positioning');
      }
    }

    return {
      issues,
      score: Math.max(0, score - issues.length * 20),
      landmarks
    };
  }

  // Analyze rowing form (generic pull exercise)
  static analyzeRowingForm(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    const score = 100;

    // Check for excessive torso movement (should be stable)
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_hip && landmarks.right_hip) {
      const shoulderCenterY = (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2;
      const hipCenterY = (landmarks.left_hip.y + landmarks.right_hip.y) / 2;
      const torsoAngle = Math.atan2(
        shoulderCenterY - hipCenterY,
        (landmarks.left_shoulder.x + landmarks.right_shoulder.x) / 2 -
        (landmarks.left_hip.x + landmarks.right_hip.x) / 2
      ) * (180 / Math.PI);

      // Torso should be relatively stable
      if (Math.abs(torsoAngle) > 15) {
        issues.push('Excessive torso movement');
      }
    }

    // Check for scapular retraction (should squeeze shoulder blades)
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_elbow && landmarks.right_elbow) {
      const shoulderWidth = this.calculateDistance(landmarks.left_shoulder, landmarks.right_shoulder);
      const elbowWidth = this.calculateDistance(landmarks.left_elbow, landmarks.right_elbow);

      // Elbows should be closer together than shoulders at peak contraction
      // This is a simplified check - in reality we'd need to detect contraction phase
      if (elbowWidth > shoulderWidth * 0.9) {
        issues.push('Limited scapular retraction');
      }
    }

    return {
      issues,
      score: Math.max(0, score - issues.length * 15),
      landmarks
    };
  }

  // Analyze push-up form
  static analyzePushUpForm(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    const score = 100;

    // Check for hip sagging (hips too low)
    if (landmarks.left_hip && landmarks.right_hip &&
        landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_ankle && landmarks.right_ankle) {
      const hipY = (landmarks.left_hip.y + landmarks.right_hip.y) / 2;
      const shoulderY = (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2;
      const ankleY = (landmarks.left_ankle.y + landmarks.right_ankle.y) / 2;

      // Body should form a straight line from shoulders to ankles
      const hipPosition = (hipY - shoulderY) / (ankleY - shoulderY);
      // Ideal hipPosition is around 0.5 (midpoint)
      if (hipPosition < 0.3 || hipPosition > 0.7) {
        issues.push('Hips sagging or piking');
      }
    }

    // Check for head position
    if (landmarks.nose && landmarks.left_ear && landmarks.right_ear) {
      const earCenterY = (landmarks.left_ear.y + landmarks.right_ear.y) / 2;
      const noseY = landmarks.nose.y;

      // Head should be neutral, not looking up or down excessively
      if (Math.abs(noseY - earCenterY) > 0.04) {
        issues.push('Head position incorrect');
      }
    }

    // Check for elbow flare
    if (landmarks.left_elbow && landmarks.right_elbow &&
        landmarks.left_shoulder && landmarks.right_shoulder) {
      // Elbows should be at roughly 45 degrees from body
      // Simplified check: elbows shouldn't be too far out to sides
      const leftElbowAngle = Math.atan2(
        landmarks.left_elbow.y - landmarks.left_shoulder.y,
        landmarks.left_elbow.x - landmarks.left_shoulder.x
      ) * (180 / Math.PI);
      const rightElbowAngle = Math.atan2(
        landmarks.right_elbow.y - landmarks.right_shoulder.y,
        landmarks.right_elbow.x - landmarks.right_shoulder.x
      ) * (180 / Math.PI);

      // Absolute angles should be reasonable (not flared out to 90 degrees)
      if (Math.abs(Math.abs(leftElbowAngle) - 45) > 30 ||
          Math.abs(Math.abs(rightElbowAngle) - 45) > 30) {
        issues.push('Elbows flared too wide');
      }
    }

    return {
      issues,
      score: Math.max(0, score - issues.length * 20),
      landmarks
    };
  }

  // Analyze lunge form
  static analyzeLungeForm(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    const score = 100;

    // Check front knee position (should not go too far past toes)
    if (landmarks.left_knee && landmarks.left_ankle &&
        landmarks.left_hip) {
      // Simplified: check if knee is significantly forward of ankle
      if (landmarks.left_knee.x < landmarks.left_ankle.x - 0.05) {
        issues.push('Front knee too far forward');
      }
    }

    // Check torso lean
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_hip && landmarks.right_hip) {
      const shoulderCenter = {
        x: (landmarks.left_shoulder.x + landmarks.right_shoulder.x) / 2,
        y: (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2
      };
      const hipCenter = {
        x: (landmarks.left_hip.x + landmarks.right_hip.x) / 2,
        y: (landmarks.left_hip.y + landmarks.right_hip.y) / 2
      };

      const torsoAngle = Math.atan2(
        shoulderCenter.y - hipCenter.y,
        shoulderCenter.x - hipCenter.x
      ) * (180 / Math.PI);

      if (Math.abs(torsoAngle) > 20) {
        issues.push('Excessive torso lean');
      }
    }

    // Check balance (hip level)
    if (landmarks.left_hip && landmarks.right_hip) {
      const hipDiff = Math.abs(landmarks.left_hip.y - landmarks.right_hip.y);
      if (hipDiff > 0.04) {
        issues.push('Uneven hip level');
      }
    }

    return {
      issues,
      score: Math.max(0, score - issues.length * 18),
      landmarks
    };
  }

  // Analyze curl form
  static analyzeCurlForm(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    const score = 100;

    // Check for excessive shoulder movement (shoulders should stay back)
    if (landmarks.left_shoulder && landmarks.right_shoulder &&
        landmarks.left_ear && landmarks.right_ear) {
      const leftShoulderForward = landmarks.left_shoulder.x - landmarks.left_ear.x;
      const rightShoulderForward = landmarks.right_shoulder.x - landmarks.right_ear.x;

      // Shoulders should not drift forward excessively
      if (leftShoulderForward > 0.03 || rightShoulderForward > 0.03) {
        issues.push('Shoulders drifting forward');
      }
    }

    // Check for elbow positioning (should stay relatively fixed)
    if (landmarks.left_elbow && landmarks.right_elbow &&
        landmarks.left_shoulder && landmarks.right_shoulder) {
      const leftElbowToShoulder = {
        x: landmarks.left_elbow.x - landmarks.left_shoulder.x,
        y: landmarks.left_elbow.y - landmarks.left_shoulder.y
      };
      const rightElbowToShoulder = {
        x: landmarks.right_elbow.x - landmarks.right_shoulder.x,
        y: landmarks.right_elbow.y - landmarks.right_shoulder.y
      };

      // Elbows should stay close to sides
      const leftElbowDistance = Math.sqrt(
        leftElbowToShoulder.x * leftElbowToShoulder.x +
        leftElbowToShoulder.y * leftElbowToShoulder.y
      );
      const rightElbowDistance = Math.sqrt(
        rightElbowToShoulder.x * rightElbowToShoulder.x +
        rightElbowToShoulder.y * rightElbowToShoulder.y
      );

      // This is a simplified check - ideal would track consistency through rep
      if (leftElbowDistance > 0.15 || rightElbowDistance > 0.15) {
        issues.push('Elbows drifting away from sides');
      }
    }

    // Check for wrist position
    if (landmarks.left_wrist && landmarks.right_wrist &&
        landmarks.left_elbow && landmarks.right_elbow) {
      // Wrists should be straight, not excessively bent back
      const leftWristAngle = Math.atan2(
        landmarks.left_wrist.y - landmarks.left_elbow.y,
        landmarks.left_wrist.x - landmarks.left_elbow.x
      ) * (180 / Math.PI);
      const rightWristAngle = Math.atan2(
        landmarks.right_wrist.y - landmarks.right_elbow.y,
        landmarks.right_wrist.x - landmarks.right_elbow.x
      ) * (180 / Math.PI);

      // Wrists should be neutral or slightly flexed, not extended back
      if (leftWristAngle > 20 || rightWristAngle > 20) {
        issues.push('Wrists extended back');
      }
    }

    return {
      issues,
      score: Math.max(0, score - issues.length * 15),
      landmarks
    };
  }

  // Generic form analysis for exercises without specific handlers
  static analyzeGenericForm(landmarks) {
    if (!landmarks) return null;

    const issues = [];
    let score = 100;

    // Basic movement symmetry check
    const symmetricPairs = [
      { left: 'left_shoulder', right: 'right_shoulder' },
      { left: 'left_elbow', right: 'right_elbow' },
      { left: 'left_wrist', right: 'right_wrist' },
      { left: 'left_hip', right: 'right_hip' },
      { left: 'left_knee', right: 'right_knee' },
      { left: 'left_ankle', right: 'right_ankle' }
    ];

    let asymmetryCount = 0;
    let totalPairs = 0;

    for (const pair of symmetricPairs) {
      if (landmarks[pair.left] && landmarks[pair.right]) {
        totalPairs++;
        // Simple asymmetry check: significant difference in position
        const diff = Math.abs(
          (landmarks[pair.left].x + landmarks[pair.left].y) / 2 -
          (landmarks[pair.right].x + landmarks[pair.right].y) / 2
        );
        if (diff > 0.08) { // Threshold for noticeable asymmetry
          asymmetryCount++;
        }
      }
    }

    if (totalPairs > 0) {
      const asymmetryRatio = asymmetryCount / totalPairs;
      if (asymmetryRatio > 0.3) { // More than 30% of pairs showing asymmetry
        issues.push('Noticeable left/right asymmetry');
        score -= 20;
      }
    }

    // Basic movement range check (simplified)
    // We could track min/max positions over time, but for now just check if limbs are moving
    const movingParts = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'];
    let movingCount = 0;

    for (const part of movingParts) {
      if (landmarks[part] && landmarks[part].visibility > 0.5) {
        movingCount++;
      }
    }

    if (movingCount < 2) {
      issues.push('Limited limb movement detected');
      score -= 15;
    }

    // Ensure score doesn't go below 0
    score = Math.max(0, score);

    return {
      issues,
      score: score,
      landmarks
    };
  }

  // Generic form analysis based on exercise type
  static analyzeForm(landmarks, exerciseType = 'squat') {
    const lowerType = exerciseType.toLowerCase();

    // Squat variations
    if (lowerType.includes('squat')) {
      return this.analyzeSquat(landmarks);
    }

    // Deadlift variations
    if (lowerType.includes('deadlift') || lowerType.includes('dl')) {
      return this.analyzeDeadlift(landmarks);
    }

    // Bench press variations
    if (lowerType.includes('bench') || lowerType.includes('bp') ||
        lowerType.includes('press') && lowerType.includes('bench')) {
      return this.analyzeBenchPress(landmarks);
    }

    // Overhead press variations
    if (lowerType.includes('overhead') || lowerType.includes('ohp') ||
        (lowerType.includes('press') && !lowerType.includes('bench'))) {
      return this.analyzeOverheadPress(landmarks);
    }

    // Rowing/pull variations
    if (lowerType.includes('row') || lowerType.includes('pull') ||
        lowerType.includes('lat') || lowerType.includes('pulldown')) {
      return this.analyzeRowingForm(landmarks);
    }

    // Push-up variations
    if (lowerType.includes('push up') || lowerType.includes('pushup')) {
      return this.analyzePushUpForm(landmarks);
    }

    // Lunge variations
    if (lowerType.includes('lunge') || lowerType.includes('split')) {
      return this.analyzeLungeForm(landmarks);
    }

    // Curl variations
    if (lowerType.includes('curl') || lowerType.includes('biceps')) {
      return this.analyzeCurlForm(landmarks);
    }

    // Default: basic movement analysis
    return this.analyzeGenericForm(landmarks);
  }
}