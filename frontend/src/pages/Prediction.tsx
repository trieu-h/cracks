import React, { useState, useRef } from 'react';
import { Upload, Play, RotateCcw } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { runPrediction } from '../api';

const Prediction: React.FC = () => {
  const [imagePath, setImagePath] = useState('');
  const [modelType, setModelType] = useState('yolo');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const fileWithPath = file as File & { path?: string };
      setImagePath(fileWithPath.path || file.name);
    }
  };

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const handlePredict = async () => {
    if (!imagePath) return;
    
    setLoading(true);
    const res = await runPrediction({
      model_path: '',
      image_path: imagePath,
      model_type: modelType,
      conf: 0.25
    });
    setResult(res.data);
    setLoading(false);
  };

  return (
    <div className="space-y-6 h-full overflow-auto">
      <div className="grid grid-cols-2 gap-6">
        {/* Configuration */}
        <Panel title="Prediction Configuration">
          <div className="space-y-4">
            {/* Model Type */}
            <div>
              <label className="text-sm text-stone-400 mb-2 block">Model Type</label>
              <div className="flex gap-2 p-1 bg-stone-800 rounded-xl">
                <button
                  onClick={() => setModelType('yolo')}
                  className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                    modelType === 'yolo'
                      ? 'bg-green-500 text-stone-950 shadow-lg'
                      : 'text-stone-400 hover:text-stone-200 hover:bg-stone-700/50'
                  }`}
                >
                  YOLO
                </button>
                <button
                  onClick={() => setModelType('rfdetr')}
                  className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                    modelType === 'rfdetr'
                      ? 'bg-green-500 text-stone-950 shadow-lg'
                      : 'text-stone-400 hover:text-stone-200 hover:bg-stone-700/50'
                  }`}
                >
                  RF-DETR
                </button>
              </div>
            </div>

            {/* Image Path */}
            <div>
              <label className="text-sm text-stone-400 mb-2 block">Image Path</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={imagePath}
                  onChange={(e) => setImagePath(e.target.value)}
                  placeholder="/path/to/image.jpg"
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
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>

            {/* Run Button */}
            <Button 
              primary 
              onClick={handlePredict} 
              disabled={loading || !imagePath}
              className="w-full mt-4 w-fit"
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
        </Panel>

        {/* Results */}
        <Panel title="Detection Results">
          {result ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <LED color={result.result.success ? 'green' : 'red'} />
                  <span className="font-mono text-green-400">
                    {result.result.success ? 'DETECTED' : 'ERROR'}
                  </span>
                </div>
                <span className="text-xs text-stone-500">
                  {(result.result.inference_time * 1000).toFixed(0)}ms
                </span>
              </div>

              {result.result.success && (
                <>
                  <div className="text-2xl font-mono text-green-400">
                    {result.result.num_detections} objects
                  </div>

                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {result.result.detections?.map((det: any, i: number) => (
                      <div 
                        key={i}
                        className="flex items-center justify-between py-2 px-3 bg-stone-800/50 rounded-lg text-sm"
                      >
                        <span className="text-stone-300">{det.class_name}</span>
                        <span className="font-mono text-green-400">
                          {(det.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-48 text-stone-500">
              <Upload size={48} className="mb-4 opacity-30" />
              <p>Configure and run prediction</p>
              <p className="text-sm mt-2">Results will appear here</p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
};

export default Prediction;
