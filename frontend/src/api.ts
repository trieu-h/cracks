import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
export const BASE_URL = API_URL.replace('/api', '');

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Datasets
export const getDatasets = () => api.get('/datasets');
export const importDataset = (path: string) => api.post('/datasets/import', { path });
export const deleteDataset = (id: string) => api.delete(`/datasets/${id}`);

// Training
export const startTraining = (config: any) => api.post('/training/start', config);
export const stopTraining = (sessionId: string) => api.post(`/training/${sessionId}/stop`);
export const getTrainingStatus = (sessionId: string) => api.get(`/training/${sessionId}/status`);
export const getTrainingMetrics = (sessionId: string) => api.get(`/training/${sessionId}/metrics`);
export const getTrainingSessions = () => api.get('/training/sessions');

// Prediction
export const runPrediction = (data: FormData) => api.post('/prediction/upload', data, {
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});
export const getPrediction = (id: string) => api.get(`/prediction/${id}`);

// Models
export const getModels = () => api.get('/models');
export const getModel = (id: string) => api.get(`/models/${id}`);
export const deleteModel = (id: string) => api.delete(`/models/${id}`);

// System
export const getGPUStats = () => api.get('/system/gpu');
export const healthCheck = () => api.get('/system/health');

export default api;
