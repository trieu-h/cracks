import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, RotateCcw, Image, Video, Camera, Download, AlertTriangle } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { runDetection, runVideoDetection, getModels, BASE_URL } from '../api';

// Detection overlay component for real-time bounding boxes
interface DetectionOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
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

const Detection: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'photo' | 'video' | 'webcam'>('photo');
  
  // Photo detection states
  const [imagePath, setImagePath] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
  // Video detection states
  const [videoPath, setVideoPath] = useState('');
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [sampleInterval, setSampleInterval] = useState<number>(5);
  
  
  // Webcam detection states
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

      
      const startTime = performance.now();
      
      try {
        const res = await runDetection(formData);
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
        console.error('Detection failed:', error);
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


      try {
        const res = await runDetection(formData);
        setResult(res.data);
      } catch (error) {
        setResult({
          success: false,
          error: 'Failed to run detection'
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

      formData.append('sample_interval', sampleInterval.toString());

      try {
        const res = await runVideoDetection(formData);
        setResult(res.data);
      } catch (error) {
        setResult({
          success: false,
          error: 'Failed to run video detection'
        });
      }
      setLoading(false);
      setProgress('');
    }
  };

  return (
    <div className="flex gap-6 h-full overflow-hidden">
      {/* Main Content Area */}
      <div className="flex-1 space-y-6 overflow-auto pr-2 pb-10 custom-scrollbar">
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
          <Panel title={`${activeTab === 'photo' ? 'Photo' : 'Video'} Detection Configuration`}>
            <div className="flex gap-4 items-start">
              {/* Model Selection */}
              <div className="w-[250px] shrink-0">
                <label className="text-sm text-stone-400 mb-2 block h-[22px]">Select Model</label>
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
              <div className="flex-1 min-w-[300px]">
                <label className="text-sm text-stone-400 mb-2 block h-[22px]">
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

              {/* Run Detection Button */}
              <div className="shrink-0">
                <label className="text-sm text-stone-400 mb-2 block h-[22px] opacity-0">Action</label>
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
                      Run Detection
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
            <div className="flex gap-4 items-start">
              {/* Model Selection */}
              <div className="w-[300px] shrink-0">
                <label className="text-sm text-stone-400 mb-2 block h-[22px]">Select Model</label>
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
                    No trained models available.
                  </p>
                )}
              </div>

              {/* Webcam Control */}
              <div className="shrink-0">
                <label className="text-sm text-stone-400 mb-2 block h-[22px]">Webcam</label>
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
              <div className="shrink-0">
                <label className="text-sm text-stone-400 mb-2 block h-[22px]">
                  {captureInterval < 300 ? '🔴 Real-time' : 'Auto'}
                </label>
                <div className="flex items-center gap-2 h-[42px]">
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
          </Panel>
        )}

        {/* Photo Detection Results */}
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

            {/* Right: Detection Result */}
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
                          <span>{result.result.num_detections} {result.result.num_detections === 1 ? 'crack' : 'cracks'} detected</span>
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
                  <p>Click "Run Detection" to see results</p>
                  <p className="text-sm mt-2">The annotated image will appear here</p>
                </div>
              )}
            </Panel>
          </div>
        )}

        {/* Video Detection Results */}
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

            {/* Right: Detection Result */}
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
                      </div>
                    </>
                  ) : (
                    <div className="flex-1 flex items-center justify-center text-red-400 min-h-0">
                      {result.result?.error || result.error || 'Video processing failed'}
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-stone-500 min-h-0">
                  <Video size={48} className="mb-4 opacity-30" />
                  <p>Click "Run Detection" to process video</p>
                  <p className="text-sm mt-2">The annotated video will appear here after processing</p>
                </div>
              )}
            </Panel>
          </div>
        )}

        {/* Webcam Results (Grid overlay when active) */}
        {activeTab === 'webcam' && webcamActive && (
          <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
            {/* Left: Live Feed with Overlay */}
            <Panel title="Live Webcam Feed" className="h-full flex flex-col relative overflow-hidden">
              <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 relative min-h-0">
                <video 
                  ref={webcamVideoRef}
                  autoPlay 
                  playsInline
                  className="max-w-full max-h-full"
                />
                <DetectionOverlay 
                  videoRef={webcamVideoRef}
                  detections={detections}
                  isActive={webcamActive}
                />
                
                {/* Stats Overlay */}
                <div className="absolute top-4 left-4 flex flex-col gap-2 z-20">
                  <div className="flex items-center gap-2 bg-stone-900/80 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-stone-700">
                    <LED color="green" pulse />
                    <span className="text-[10px] font-mono text-stone-300">LIVE FEED</span>
                  </div>
                  {fps > 0 && (
                    <div className="bg-stone-900/80 backdrop-blur-sm px-3 py-1 rounded-lg border border-stone-700 text-[10px] font-mono text-green-400">
                      {fps} FPS
                    </div>
                  )}
                </div>
              </div>

              {/* Controls */}
              <div className="mt-4 flex gap-3 shrink-0">
                <Button 
                  primary 
                  className="flex-1"
                  onClick={captureFromWebcam}
                  disabled={isCapturing || !selectedModel}
                >
                  <Camera size={18} className="inline mr-2" />
                  {isCapturing ? 'Processing...' : 'Capture & Detect'}
                </Button>
                <div className="flex items-center gap-2 bg-stone-800/50 px-4 rounded-xl border border-stone-700">
                  <span className="text-xs text-stone-500 uppercase font-bold tracking-wider">Auto</span>
                  <input
                    type="checkbox"
                    checked={autoCapture}
                    onChange={(e) => setAutoCapture(e.target.checked)}
                    className="rounded bg-stone-700 border-stone-600 w-4 h-4"
                  />
                </div>
              </div>
            </Panel>

            {/* Right: Detected Objects Column */}
            <Panel title="Detection Stream" className="h-full flex flex-col">
              {webcamResult ? (
                <div className="h-full flex flex-col min-h-0">
                  <div className="flex items-center justify-between mb-4 shrink-0">
                    <div className="flex items-center gap-2">
                      <LED color={webcamResult.success && webcamResult.result?.success ? 'green' : 'red'} />
                      <span className={`font-mono ${webcamResult.success && webcamResult.result?.success ? 'text-green-400' : 'text-red-400'}`}>
                        {webcamResult.success && webcamResult.result?.success ? 'LAST RESULT' : 'ERROR'}
                      </span>
                    </div>
                  </div>

                  {webcamResult.success && webcamResult.result?.success ? (
                    <>
                      {webcamResult.result.annotated_image ? (
                        <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 relative min-h-0">
                          <img 
                            src={`${BASE_URL}/predictions/${webcamResult.result.annotated_image.replace(/\\/g, '/').split('/').pop()}`}
                            alt="Webcam Result"
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
    </div>
  );
};

export default Detection;
