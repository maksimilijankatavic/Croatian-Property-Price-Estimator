/**
 * API service for communicating with the HF Spaces backend.
 */

import type { PredictionInput, PredictionResult } from './types';

const HF_SPACES_URL = process.env.NEXT_PUBLIC_HF_SPACES_URL || 'https://your-username-croatian-property-estimator.hf.space';

/**
 * Predict property price using the V3 LightGBM model.
 */
export async function predictPrice(input: PredictionInput): Promise<PredictionResult> {
  const response = await fetch(`${HF_SPACES_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Nepoznata greška' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Check if the backend is ready (useful for handling cold starts).
 */
export async function checkHealth(): Promise<{ ready: boolean; message?: string }> {
  try {
    const response = await fetch(`${HF_SPACES_URL}/health`, {
      method: 'GET',
    });

    if (response.ok) {
      return { ready: true };
    }

    return { ready: false, message: 'Backend nije spreman' };
  } catch {
    return { ready: false, message: 'Backend se pokreće...' };
  }
}
