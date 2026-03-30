import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, RotateCcw, Image, Video, Camera, Download, AlertTriangle } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { runPrediction, runVideoPrediction, getModels, BASE_URL } from '../api';

// Detection overlay component for real-time bounding boxes
interface DetectionOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  detections: any[];
  isActive: boolean;
}

const DetectionOverlay: React.FC<DetectionOverlayProps> = ({ videoRef, detections, isActive }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!isActive || !canvasRef.current || !videoRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Match canvas size to video
    const resizeCanvas = () => {
      const rect = video.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [isActive, videoRef]);
  
  useEffect(() => {
    if (!canvasRef.current || !videoRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (detections.length === 0) return;
    
    // Calculate scale factors
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    // Draw bounding boxes
    detections.forEach((det) => {
      const bbox = det.bbox;
      if (!bbox || bbox.length < 4) return;
      
      const x = bbox[0] * scaleX;
      const y = bbox[1] * scaleY;
      const width = (bbox[2] - bbox[0]) * scaleX;
      const height = (bbox[3] - bbox[1]) * scaleY;
      
      // Draw box
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);
      
      // Draw label background
      const label = `${det.class_name || 'crack'} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = 'bold 14px sans-serif';
      const textMetrics = ctx.measureText(label);
      const textHeight = 20;
      
      ctx.fillStyle = '#22c55e';
      ctx.fillRect(x, y - textHeight, textMetrics.width + 10, textHeight);
      
      // Draw label text
      ctx.fillStyle = '#000';
      ctx.fillText(label, x + 5, y - 5);
    });
  }, [detections, videoRef]);
  
  if (!isActive) return null;
  
  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none z-10"
      style={{ width: '100%', height: '100%' }}
    />
  );
};

const Prediction: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'photo' | 'video' | 'webcam'>('photo');
  
  // Photo prediction states
  const [imagePath, setImagePath] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
  // Video prediction states
  const [videoPath, setVideoPath] = useState('');
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [sampleInterval, setSampleInterval] = useState<number>(5);
  
  // Confidence threshold state (0-100)
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(50);
  
  // Webcam prediction states
  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [autoCapture, setAutoCapture] = useState(false);
  const [captureInterval, setCaptureInterval] = useState<number>(500);
  const [webcamResult, setWebcamResult] = useState<any>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [fps, setFps] = useState<number>(0);
  
  // Common states
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [models, setModels] = useState<any[]>([]);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<string>('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const webcamVideoRef = useRef<HTMLVideoElement>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement>(null);
  const autoCaptureRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    getModels().then(res => setModels(res.data));
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setImagePath(file.name);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleVideoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedVideo(file);
      setVideoPath(file.name);
      setVideoPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const openFilePicker = () => {
    if (activeTab === 'photo') {
      fileInputRef.current?.click();
    } else if (activeTab === 'video') {
      videoInputRef.current?.click();
    }
  };

  // Webcam functions
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false 
      });
      setWebcamStream(stream);
      setWebcamActive(true);
    } catch (err) {
      console.error('Error accessing webcam:', err);
      alert('Could not access webcam. Please ensure you have granted camera permissions.');
    }
  };

  // Effect to connect video element to stream
  useEffect(() => {
    if (webcamActive && webcamStream && webcamVideoRef.current) {
      webcamVideoRef.current.srcObject = webcamStream;
    }
  }, [webcamActive, webcamStream]);

  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
    }
    setWebcamActive(false);
    if (autoCaptureRef.current) {
      clearInterval(autoCaptureRef.current);
      autoCaptureRef.current = null;
    }
    setAutoCapture(false);
    setWebcamResult(null);
  };

  const captureFromWebcam = async () => {
    if (!webcamVideoRef.current || !selectedModel || isCapturing) return;
    
    setIsCapturing(true);
    const video = webcamVideoRef.current;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (!blob) {
        setIsCapturing(false);
        return;
      }
      
      const selectedModelData = models.find(m => m.id === selectedModel);
      const file = new File([blob], `webcam_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
      
      const formData = new FormData();
      formData.append('image', file);
      formData.append('model_path', selectedModelData?.path || '');
      formData.append('conf', (confidenceThreshold / 100).toString());
      
      const startTime = performance.now();
      
      try {
        const res = await runPrediction(formData);
        setWebcamResult(res.data);
        
        // Update detections for overlay
        if (res.data?.success && res.data?.result?.success) {
          setDetections(res.data.result.detections || []);
          
          // Calculate FPS
          const endTime = performance.now();
          const inferenceTime = endTime - startTime;
          setFps(Math.round(1000 / inferenceTime));
        }
      } catch (error) {
        console.error('Prediction failed:', error);
      } finally {
        setIsCapturing(false);
      }
    }, 'image/jpeg', 0.8);
  };

  // Auto-capture effect
  useEffect(() => {
    if (autoCapture && webcamActive && selectedModel) {
      autoCaptureRef.current = setInterval(() => {
        captureFromWebcam();
      }, captureInterval);
    } else if (autoCaptureRef.current) {
      clearInterval(autoCaptureRef.current);
      autoCaptureRef.current = null;
    }
    
    return () => {
      if (autoCaptureRef.current) {
        clearInterval(autoCaptureRef.current);
        autoCaptureRef.current = null;
      }
    };
  }, [autoCapture, webcamActive, selectedModel, captureInterval]);

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  const handlePredict = async () => {
    const selectedModelData = models.find(m => m.id === selectedModel);
    
    if (activeTab === 'photo') {
      if (!selectedFile || !selectedModel) return;

      setLoading(true);
      setProgress('');

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('model_path', selectedModelData?.path || '');
      formData.append('conf', (confidenceThreshold / 100).toString());

      try {
        const res = await runPrediction(formData);
        setResult(res.data);
      } catch (error) {
        setResult({
          success: false,
          error: 'Failed to run prediction'
        });
      }
      setLoading(false);
    } else if (activeTab === 'video') {
      if (!selectedVideo || !selectedModel) return;

      setLoading(true);
      setProgress('Extracting frames... This may take a while.');

      const formData = new FormData();
      formData.append('video', selectedVideo);
      formData.append('model_path', selectedModelData?.path || '');
      formData.append('conf', (confidenceThreshold / 100).toString());
      formData.append('sample_interval', sampleInterval.toString());

      try {
        const res = await runVideoPrediction(formData);
        setResult(res.data);
      } catch (error) {
        setResult({
          success: false,
          error: 'Failed to run video prediction'
        });
      }
      setLoading(false);
      setProgress('');
    }
  };

  return (
    <div className="space-y-6 h-full overflow-auto">
      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-stone-700 pb-4">
        <button
          onClick={() => {
            setActiveTab('photo');
            stopWebcam();
          }}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'photo' 
              ? 'bg-stone-700 text-white' 
              : 'text-stone-400 hover:text-stone-200 hover:bg-stone-800'
          }`}
        >
          <Image size={18} />
          Photo
        </button>
        <button
          onClick={() => {
            setActiveTab('video');
            stopWebcam();
          }}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'video' 
              ? 'bg-stone-700 text-white' 
              : 'text-stone-400 hover:text-stone-200 hover:bg-stone-800'
          }`}
        >
          <Video size={18} />
          Video
        </button>
        <button
          onClick={() => setActiveTab('webcam')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'webcam' 
              ? 'bg-stone-700 text-white' 
              : 'text-stone-400 hover:text-stone-200 hover:bg-stone-800'
          }`}
        >
          <Camera size={18} />
          Webcam
        </button>
      </div>

      {/* Configuration Panel */}
      {activeTab !== 'webcam' && (
        <Panel title={`${activeTab === 'photo' ? 'Photo' : 'Video'} Prediction Configuration`}>
          <div className="flex items-end gap-4 flex-wrap">
            {/* Model Selection */}
            <div style={{ minWidth: '250px', flex: '1 1 250px' }}>
              <label className="text-sm text-stone-400 mb-2 block">Select Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input-clean w-full"
              >
                <option value="">Choose a trained model...</option>
                {models.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.id}) - {(model.size / (1024 * 1024)).toFixed(1)} MB
                  </option>
                ))}
              </select>
              {models.length === 0 && (
                <p className="text-xs text-stone-500 mt-2">
                  No trained models available. Train a model first.
                </p>
              )}
            </div>

            {/* File Upload */}
            <div style={{ minWidth: '300px', flex: '2 1 300px' }}>
              <label className="text-sm text-stone-400 mb-2 block">
                Select {activeTab === 'photo' ? 'Image' : 'Video'}
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={activeTab === 'photo' ? imagePath : videoPath}
                  readOnly
                  placeholder={`Click Browse to select a ${activeTab}...`}
                  className="input-clean flex-1"
                />
                <button
                  onClick={openFilePicker}
                  className="px-4 py-2 bg-stone-700 hover:bg-stone-600 rounded-xl text-sm text-stone-200 transition-all"
                  title="Browse files"
                >
                  Browse
                </button>
              </div>
            </div>

            {/* Confidence Threshold Slider */}
            <div style={{ minWidth: '200px', flex: '1 1 200px' }}>
              <label className="text-sm text-stone-400 mb-2 block">Confidence Threshold</label>
              <input
                type="range"
                min="0"
                max="100"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-stone-500 mt-1">
                <span>0%</span>
                <span>{confidenceThreshold}%</span>
                <span>100%</span>
              </div>
            </div>

            {/* Run Prediction Button */}
            <div>
              <label className="text-sm text-stone-400 mb-2 block opacity-0">Action</label>
              <Button 
                primary 
                onClick={handlePredict} 
                disabled={loading || !(activeTab === 'photo' ? selectedFile : selectedVideo) || !selectedModel}
              >
                {loading ? (
                  <>
                    <RotateCcw size={18} className="inline mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play size={18} className="inline mr-2" />
                    Run Prediction
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Secondary Controls Row */}
          <div className="flex gap-4 mt-4 flex-wrap">
            {activeTab === 'video' && (
              <div style={{ minWidth: '300px', flex: '1 1 300px' }}>
                <label className="text-xs text-stone-400">Frame Sample Interval (lower = more accurate but slower)</label>
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={sampleInterval}
                  onChange={(e) => setSampleInterval(parseInt(e.target.value))}
                  className="w-full mt-1"
                />
                <div className="flex justify-between text-xs text-stone-500 mt-1">
                  <span>1 (all frames)</span>
                  <span>{sampleInterval} frame(s)</span>
                  <span>30 (fast)</span>
                </div>
              </div>
            )}
            {progress && (
              <p className="text-xs text-blue-400 mt-2 animate-pulse">{progress}</p>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          <input
            ref={videoInputRef}
            type="file"
            accept="video/*"
            onChange={handleVideoSelect}
            className="hidden"
          />
        </Panel>
      )}

      {/* Webcam Configuration Panel */}
      {activeTab === 'webcam' && (
        <Panel title="Webcam Detection Configuration">
          <div className="flex items-end gap-4">
            {/* Model Selection */}
            <div style={{ width: '500px' }}>
              <label className="text-sm text-stone-400 mb-1 block">Select Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input-clean w-full"
              >
                <option value="">Choose a trained model...</option>
                {models.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.id}) - {(model.size / (1024 * 1024)).toFixed(1)} MB
                  </option>
                ))}
              </select>
              {models.length === 0 && (
                <p className="text-xs text-stone-500 mt-1">
                  No trained models available.
                </p>
              )}
            </div>

            {/* Webcam Control */}
            <div>
              <label className="text-sm text-stone-400 mb-1 block">Webcam</label>
              <div className="flex">
                {!webcamActive ? (
                  <Button
                    primary
                    onClick={startWebcam}
                  >
                    <Camera size={18} className="inline mr-2" />
                    Start
                  </Button>
                ) : (
                  <Button
                    onClick={stopWebcam}
                    className="bg-red-600 hover:bg-red-700"
                  >
                    Stop
                  </Button>
                )}
              </div>
            </div>

            {/* Auto-Capture */}
            <div className="flex items-center gap-2">
              <div className="flex flex-col">
                <label className="text-sm text-stone-400 mb-1 block">
                  {captureInterval < 300 ? '🔴 Real-time' : 'Auto'}
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={autoCapture}
                    onChange={(e) => setAutoCapture(e.target.checked)}
                    disabled={!webcamActive}
                    className="rounded bg-stone-700 border-stone-600"
                  />
                  <input
                    type="number"
                    min="100"
                    max="5000"
                    step="100"
                    value={captureInterval}
                    onChange={(e) => setCaptureInterval(parseInt(e.target.value))}
                    disabled={!webcamActive}
                    className="input-clean w-14 text-center"
                  />
                  <span className="text-sm text-stone-400">ms</span>
                </div>
              </div>
            </div>

            {/* Confidence Threshold Slider */}
            <div className="flex flex-col" style={{ width: '200px' }}>
              <label className="text-sm text-stone-400 mb-1 block">Confidence Threshold</label>
              <input
                type="range"
                min="0"
                max="100"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-stone-500 mt-1">
                <span>0%</span>
                <span>{confidenceThreshold}%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        </Panel>
      )}

      {/* Photo Prediction Results */}
      {activeTab === 'photo' && imagePreview && (
        <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
          {/* Left: Original Image */}
          <Panel title="Original Image" className="h-full flex flex-col">
            <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0">
              <img 
                src={imagePreview}
                alt="Original"
                className="max-w-full max-h-full object-contain"
              />
            </div>
          </Panel>

          {/* Right: Prediction Result */}
          <Panel title="Detection Results" className="h-full flex flex-col">
            {result ? (
              <div className="h-full flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <div className="flex items-center gap-2">
                    <LED color={result.success && result.result?.success ? 'green' : 'red'} />
                    <span className={`font-mono ${result.success && result.result?.success ? 'text-green-400' : 'text-red-400'}`}>
                      {result.success && result.result?.success ? 'SEGMENTED' : 'ERROR'}
                    </span>
                  </div>
                  {result.result?.inference_time && (
                    <span className="text-xs text-stone-500">
                      {(result.result.inference_time * 1000).toFixed(0)}ms
                    </span>
                  )}
                </div>

                {result.success && result.result?.success ? (
                  <>
                    {result.result.annotated_image ? (
                      <div className="flex-1 flex flex-col min-h-0 relative group">
                        <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 relative">
                          <img 
                            src={`${BASE_URL}/predictions/${result.result.annotated_image.replace(/\\/g, '/').split('/').pop()}`}
                            alt="Detection Result"
                            className="max-w-full max-h-full object-contain"
                            onError={(e) => {
                              console.error('Failed to load image:', e);
                              (e.target as HTMLImageElement).style.display = 'none';
                            }}
                          />
                          <a 
                            href={`${BASE_URL}/predictions/${result.result.annotated_image.replace(/\\/g, '/').split('/').pop()}`}
                            download={`predicted_${result.result.annotated_image.replace(/\\/g, '/').split('/').pop()}`}
                            className="absolute bottom-4 right-4 bg-stone-800/80 hover:bg-stone-700 text-stone-200 p-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 backdrop-blur-sm shadow-lg border border-stone-600"
                            title="Download Image"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            <Download size={16} />
                            <span className="text-xs font-medium">Download</span>
                          </a>
                        </div>
                      </div>
                    ) : (
                      <div className="flex-1 flex items-center justify-center text-stone-500 min-h-0">
                        No annotated image available
                      </div>
                    )}

                    <div className="mt-4 shrink-0">
                      <div className="flex items-center justify-between text-sm text-stone-400 mb-2">
                        <span>{result.result.num_detections} objects detected</span>
                        <span className="text-xs text-blue-400 bg-blue-900/20 px-2 py-1 rounded">Analysis</span>
                      </div>
                      
                      {result.result.num_detections > 0 && result.result.detections && (
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="bg-stone-800/50 p-2 rounded flex flex-col">
                            <span className="text-stone-500">Average Confidence:</span>
                            <span className="text-stone-300">
                              {(result.result.detections.reduce((acc: number, d: any) => acc + d.confidence, 0) / result.result.num_detections * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="bg-stone-800/50 p-2 rounded flex flex-col">
                            <span className="text-stone-500">Total Crack Area:</span>
                            <span className="text-stone-300">
                              {result.result.detections.reduce((acc: number, d: any) => acc + (d.area || 0), 0).toLocaleString(undefined, { maximumFractionDigits: 0 })} px²
                            </span>
                          </div>
                          <div className="bg-stone-800/50 p-2 rounded flex flex-col col-span-2">
                            <span className="text-stone-500 mb-1">Severity Breakdown:</span>
                            <div className="flex gap-4">
                              <span className="text-red-400 flex items-center gap-1">
                                <AlertTriangle size={10} />
                                {result.result.detections.filter((d: any) => d.severity === 'High').length} High
                              </span>
                              <span className="text-yellow-400">
                                {result.result.detections.filter((d: any) => d.severity === 'Medium').length} Medium
                              </span>
                              <span className="text-stone-400">
                                {result.result.detections.filter((d: any) => d.severity === 'Low').length} Low
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-red-400 min-h-0">
                    {result.result?.error || result.error || 'An unknown error occurred'}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-stone-500 min-h-0">
                <Upload size={48} className="mb-4 opacity-30" />
                <p>Click "Run Prediction" to see results</p>
                <p className="text-sm mt-2">The annotated image will appear here</p>
              </div>
            )}
          </Panel>
        </div>
      )}

      {/* Video Prediction Results */}
      {activeTab === 'video' && videoPreview && (
        <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
          {/* Left: Original Video */}
          <Panel title="Original Video" className="h-full flex flex-col">
            <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0">
              <video 
                src={videoPreview}
                controls
                className="max-w-full max-h-full"
              />
            </div>
          </Panel>

          {/* Right: Prediction Result */}
          <Panel title="Detection Results" className="h-full flex flex-col">
            {result ? (
              <div className="h-full flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <div className="flex items-center gap-2">
                    <LED color={result.success && result.result?.success ? 'green' : 'red'} />
                    <span className={`font-mono ${result.success && result.result?.success ? 'text-green-400' : 'text-red-400'}`}>
                      {result.success && result.result?.success ? 'PROCESSED' : 'ERROR'}
                    </span>
                  </div>
                </div>

                {result.success && result.result?.success ? (
                  <>
                    {result.result.annotated_video ? (
                      <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0 relative group">
                        <video 
                          src={`${BASE_URL}/predictions/${result.result.annotated_video.replace(/\\/g, '/').split('/').pop()}`}
                          controls
                          className="max-w-full max-h-full"
                        />
                        <a 
                          href={`${BASE_URL}/predictions/${result.result.annotated_video.replace(/\\/g, '/').split('/').pop()}`}
                          download={`predicted_${result.result.annotated_video.replace(/\\/g, '/').split('/').pop()}`}
                          className="absolute bottom-4 right-4 bg-stone-800/80 hover:bg-stone-700 text-stone-200 p-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 backdrop-blur-sm shadow-lg border border-stone-600 z-10"
                          title="Download Video"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <Download size={16} />
                          <span className="text-xs font-medium">Download</span>
                        </a>
                      </div>
                    ) : (
                      <div className="flex-1 flex items-center justify-center text-stone-500 min-h-0">
                        No annotated video available
                      </div>
                    )}

                    <div className="mt-4 space-y-2 text-sm text-stone-400 shrink-0">
                      <div className="flex items-center justify-between">
                        <span>Total detections: {result.result.total_detections}</span>
                        <span className="text-xs text-blue-400 bg-blue-900/20 px-2 py-1 rounded">Video Segmentation</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="bg-stone-800/50 p-2 rounded">
                          <span className="text-stone-500">Frames processed:</span>
                          <span className="ml-2 text-stone-300">{result.result.processed_frames}/{result.result.total_frames}</span>
                        </div>
                        <div className="bg-stone-800/50 p-2 rounded">
                          <span className="text-stone-500">Frames with cracks:</span>
                          <span className="ml-2 text-stone-300">{result.result.frames_with_detections}</span>
                        </div>
                        <div className="bg-stone-800/50 p-2 rounded">
                          <span className="text-stone-500">Sample interval:</span>
                          <span className="ml-2 text-stone-300">{result.result.sample_interval}</span>
                        </div>
                        <div className="bg-stone-800/50 p-2 rounded">
                          <span className="text-stone-500">Processing time:</span>
                          <span className="ml-2 text-stone-300">{result.result.total_time?.toFixed(1)}s</span>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-red-400 min-h-0">
                    {result.result?.error || result.error || 'An unknown error occurred'}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-stone-500 min-h-0">
                <Video size={48} className="mb-4 opacity-30" />
                <p>Click "Run Prediction" to see results</p>
                <p className="text-sm mt-2">The annotated video will appear here</p>
              </div>
            )}
          </Panel>
        </div>
      )}

      {/* Empty State for Photo */}
      {activeTab === 'photo' && !imagePreview && (
        <div className="flex flex-col items-center justify-center h-96 text-stone-500">
          <Upload size={64} className="mb-4 opacity-30" />
          <p className="text-lg">Select an image to start</p>
          <p className="text-sm mt-2">Choose a photo and click "Browse" to get started</p>
        </div>
      )}

      {/* Empty State for Video */}
      {activeTab === 'video' && !videoPreview && (
        <div className="flex flex-col items-center justify-center h-96 text-stone-500">
          <Video size={64} className="mb-4 opacity-30" />
          <p className="text-lg">Select a video to start</p>
          <p className="text-sm mt-2">Choose a video and click "Browse" to get started</p>
        </div>
      )}

      {/* Webcam Detection */}
      {activeTab === 'webcam' && (
        <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
          {/* Left: Live Webcam Feed */}
          <Panel 
            title={`Live Webcam ${fps > 0 ? `(${fps} FPS)` : ''}`} 
            className="h-full flex flex-col"
          >
            <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0 relative">
              {webcamActive ? (
                <>
                  <video
                    ref={webcamVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />
                  {/* Overlay canvas for real-time bounding boxes */}
                  <DetectionOverlay 
                    videoRef={webcamVideoRef}
                    detections={detections}
                    isActive={webcamActive}
                  />
                </>
              ) : (
                <div className="flex flex-col items-center justify-center text-stone-500">
                  <Camera size={64} className="mb-4 opacity-30" />
                  <p>Webcam is off</p>
                  <p className="text-sm mt-2">Click "Start Webcam" to begin</p>
                </div>
              )}
              
              {/* Hidden canvas for frame capture */}
              <canvas ref={webcamCanvasRef} className="hidden" />
            </div>
            
            {/* Capture Button */}
            {webcamActive && (
              <div className="mt-4 flex justify-center gap-2">
                {!autoCapture && (
                  <Button
                    primary
                    onClick={captureFromWebcam}
                    disabled={isCapturing || !selectedModel}
                  >
                    {isCapturing ? (
                      <>
                        <RotateCcw size={18} className="inline mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Camera size={18} className="inline mr-2" />
                        Capture & Detect
                      </>
                    )}
                  </Button>
                )}
                {autoCapture && (
                  <div className="flex items-center gap-2 px-4 py-2 bg-green-900/30 rounded-lg">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-sm text-green-400">
                      Real-time detection active ({captureInterval}ms)
                    </span>
                  </div>
                )}
              </div>
            )}
          </Panel>

          {/* Right: Detection Result */}
          <Panel title="Detection Results" className="h-full flex flex-col">
            {webcamResult ? (
              <div className="h-full flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <div className="flex items-center gap-2">
                    <LED color={webcamResult.success && webcamResult.result?.success ? 'green' : 'red'} />
                    <span className={`font-mono ${webcamResult.success && webcamResult.result?.success ? 'text-green-400' : 'text-red-400'}`}>
                      {webcamResult.success && webcamResult.result?.success ? 'DETECTED' : 'ERROR'}
                    </span>
                  </div>
                  {webcamResult.result?.inference_time && (
                    <span className="text-xs text-stone-500">
                      {(webcamResult.result.inference_time * 1000).toFixed(0)}ms
                    </span>
                  )}
                </div>

                {webcamResult.success && webcamResult.result?.success ? (
                  <>
                    {webcamResult.result.annotated_image ? (
                      <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0">
                        <img
                          src={`${BASE_URL}/predictions/${webcamResult.result.annotated_image.split('/').pop()}`}
                          alt="Detection Result"
                          className="max-w-full max-h-full object-contain"
                          onError={(e) => {
                            console.error('Failed to load image:', e);
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                      </div>
                    ) : (
                      <div className="flex-1 flex items-center justify-center text-stone-500 min-h-0">
                        No detection result available
                      </div>
                    )}

                    <div className="mt-4 text-sm text-stone-400 shrink-0 flex items-center justify-between">
                      <span>{webcamResult.result.num_detections} cracks detected</span>
                      <span className="text-xs text-blue-400 bg-blue-900/20 px-2 py-1 rounded">
                        {autoCapture ? 'Auto-capturing' : 'Manual capture'}
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-red-400 min-h-0">
                    {webcamResult.result?.error || webcamResult.error || 'Detection failed'}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-stone-500 min-h-0">
                <Camera size={48} className="mb-4 opacity-30" />
                <p>Ready to detect cracks</p>
                <p className="text-sm mt-2">
                  {autoCapture 
                    ? captureInterval < 300 
                      ? '🔴 Real-time mode: Detections appear with bounding boxes on live feed'
                      : 'Auto-capturing enabled - results appear here with live overlay'
                    : 'Click "Capture & Detect" or enable auto-capture for real-time detection'
                  }
                </p>
              </div>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
};

export default Prediction;
