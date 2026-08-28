// Pose estimation service using MediaPipe
import * as pose from '@mediapipe/pose';

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

    this.pose = new pose.Pose({
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
    this.pose.reset();
  }

  close() {
    this.pose.close();
  }
}

// Utility functions for form analysis
export class FormAnalyzer {
  // Calculate angle between three points
  static calculateAngle(a, b, c) {
    const ab = { x: b.x - a.x, y: b.y - a.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };

    const dotProduct = ab.x * bc.x + ab.y * bc.y;
    const magnitudeAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
    const magnitudeBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);

    const angle = Math.acos(dotProduct / (magnitudeAB * magnitudeBC));
    return angle * (180 / Math.PI); // Convert to degrees
  }

  // Calculate distance between two points
  static calculateDistance(a, b) {
    return Math.sqrt(
      Math.pow(b.x - a.x, 2) +
      Math.pow(b.y - a.y, 2) +
      Math.pow(b.z - a.z, 2)
    );
  }

  // Normalize landmarks relative to hip center
  static normalizeLandmarks(landmarks) {
    if (!landmarks.left_hip || !landmarks.right_hip) return landmarks;

    const hipCenter = {
      x: (landmarks.left_hip.x + landmarks.right_hip.x) / 2,
      y: (landmarks.left_hip.y + landmarks.right_hip.y) / 2,
      z: (landmarks.left_hip.z + landmarks.right_hip.z) / 2
    };

    const normalized = {};
    for (const [key, point] of Object.entries(landmarks)) {
      normalized[key] = {
        x: point.x - hipCenter.x,
        y: point.y - hipCenter.y,
        z: point.z - hipCenter.z,
        visibility: point.visibility
      };
    }
    return normalized;
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

  // Generic form analysis based on exercise type
  static analyzeForm(landmarks, exerciseType = 'squat') {
    switch (exerciseType.toLowerCase()) {
      case 'squat':
        return this.analyzeSquat(landmarks);
      case 'deadlift':
        return this.analyzeDeadlift(landmarks);
      case 'bench press':
        // TODO: Implement bench press analysis
        return { issues: [], score: 100, landmarks };
      default:
        return { issues: [], score: 100, landmarks };
    }
  }
}