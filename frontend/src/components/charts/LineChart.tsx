import React, { useState } from 'react';

interface LineChartProps {
  data: {
    epoch: number;
    f1?: number;
    precision?: number;
    recall?: number;
    mAP50?: number;
    mAP50_95?: number;
  }[];
  width?: number;
  height?: number;
}

const LineChart: React.FC<LineChartProps> = ({ 
  data, 
  width = 800, 
  height = 350 
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<{
    epoch: number;
    metric: string;
    value: number;
    x: number;
    y: number;
  } | null>(null);

  if (data.length === 0) return (
    <div 
      className="flex items-center justify-center h-64 rounded-lg border border-dashed"
      style={{ borderColor: 'var(--border-primary)', color: 'var(--text-muted)' }}
    >
      No training data available
    </div>
  );

  // Chart configuration
  const padding = { top: 40, right: 80, bottom: 60, left: 60 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Metrics configuration with colors
  const metrics = [
    { key: 'f1', label: 'F1 Score', color: '#8b5cf6' },
    { key: 'precision', label: 'Precision', color: '#3b82f6' },
    { key: 'recall', label: 'Recall', color: '#ec4899' },
    { key: 'mAP50', label: 'mAP50', color: 'var(--success-text)' },
    { key: 'mAP50_95', label: 'mAP50-95', color: '#f59e0b' }
  ];

  // Normalize data - add epoch if missing (use index + 1)
  const normalizedData = data.map((d, i) => ({
    ...d,
    epoch: d.epoch !== undefined ? d.epoch : i + 1
  }));

  // Show message if only final metrics available (single data point)
  const hasOnlyFinalMetrics = normalizedData.length === 1;

  // Calculate scales
  const maxEpoch = Math.max(...normalizedData.map(d => d.epoch), 1);
  const minEpoch = Math.min(...normalizedData.map(d => d.epoch), 1);
  const epochRange = maxEpoch - minEpoch;
  const xScale = epochRange > 0 ? chartWidth / epochRange : chartWidth / 2;
  const yScale = chartHeight / 100; // Metrics are 0-1, displayed as 0-100%
  
  // Helper to calculate x position
  const getX = (epoch: number) => {
    if (epochRange === 0) return padding.left + chartWidth / 2; // Center for single point
    return padding.left + (epoch - minEpoch) * xScale;
  };

  // Generate grid lines
  const yTicks = [0, 25, 50, 75, 100];
  const xTicks = epochRange > 0 
    ? Array.from({ length: Math.min(epochRange + 1, 11) }, (_, i) => 
        Math.round(minEpoch + (epochRange / Math.min(epochRange, 10)) * i)
      ).filter((v, i, arr) => i === 0 || v !== arr[i-1])
    : [minEpoch];

  // Generate line paths
  const getPath = (metricKey: string) => {
    const points = normalizedData
      .filter(d => d[metricKey as keyof typeof d] !== undefined && d[metricKey as keyof typeof d] !== null)
      .map(d => ({
        x: getX(d.epoch),
        y: padding.top + chartHeight - ((d[metricKey as keyof typeof d] as number) * 100 * yScale)
      }));
    
    if (points.length < 2) return '';
    
    return points.reduce((path, point, i) => {
      if (i === 0) return `M ${point.x} ${point.y}`;
      return `${path} L ${point.x} ${point.y}`;
    }, '');
  };

  // Handle hover
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - padding.left;
    
    if (x >= 0 && x <= chartWidth) {
      // Convert x position back to epoch
      let epoch: number;
      if (epochRange === 0) {
        epoch = minEpoch;
      } else {
        epoch = Math.round(minEpoch + (x / xScale));
      }
      const dataPoint = normalizedData.find(d => d.epoch === epoch);
      
      if (dataPoint) {
        // Find closest metric at this position
        const mouseY = e.clientY - rect.top;
        let closestMetric: { metric: string; value: number; x: number; y: number } | null = null;
        let minDistance = Infinity;
        
        metrics.forEach(m => {
          const value = dataPoint[m.key as keyof typeof dataPoint] as number | undefined;
          if (value !== undefined && value !== null) {
            const y = padding.top + chartHeight - (value * 100 * yScale);
            const distance = Math.abs(mouseY - y);
            if (distance < 20 && distance < minDistance) {
              minDistance = distance;
              closestMetric = { metric: m.label, value, x: getX(epoch), y };
            }
          }
        });
        
        if (closestMetric) {
          setHoveredPoint({ 
            epoch, 
            metric: (closestMetric as any).metric, 
            value: (closestMetric as any).value, 
            x: (closestMetric as any).x, 
            y: (closestMetric as any).y 
          });
        } else {
          setHoveredPoint(null);
        }
      }
    } else {
      setHoveredPoint(null);
    }
  };

  return (
    <div className="w-full" style={{ minWidth: width }}>
      <svg 
        width="100%" 
        height={height} 
        viewBox={`0 0 ${width} ${height}`}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredPoint(null)}
        style={{ overflow: 'visible' }}
      >
        {/* Grid lines - Y axis */}
        {yTicks.map(tick => (
          <g key={`y-grid-${tick}`}>
            <line
              x1={padding.left}
              y1={padding.top + chartHeight - (tick * yScale)}
              x2={padding.left + chartWidth}
              y2={padding.top + chartHeight - (tick * yScale)}
              stroke="var(--border-primary)"
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.5}
            />
            <text
              x={padding.left - 10}
              y={padding.top + chartHeight - (tick * yScale) + 4}
              textAnchor="end"
              fontSize={11}
              fill="var(--text-muted)"
            >
              {tick}%
            </text>
          </g>
        ))}

        {/* Grid lines - X axis */}
        {xTicks.map(tick => (
          <g key={`x-grid-${tick}`}>
            <line
              x1={getX(tick)}
              y1={padding.top}
              x2={getX(tick)}
              y2={padding.top + chartHeight}
              stroke="var(--border-primary)"
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.3}
            />
            <text
              x={getX(tick)}
              y={padding.top + chartHeight + 20}
              textAnchor="middle"
              fontSize={11}
              fill="var(--text-muted)"
            >
              {tick}
            </text>
          </g>
        ))}

        {/* Axis lines */}
        <line
          x1={padding.left}
          y1={padding.top + chartHeight}
          x2={padding.left + chartWidth}
          y2={padding.top + chartHeight}
          stroke="var(--border-primary)"
          strokeWidth={2}
        />
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={padding.top + chartHeight}
          stroke="var(--border-primary)"
          strokeWidth={2}
        />

        {/* Axis labels */}
        <text
          x={padding.left + chartWidth / 2}
          y={height - 15}
          textAnchor="middle"
          fontSize={12}
          fill="var(--text-secondary)"
          fontWeight={500}
        >
          Training Epochs
        </text>
        <text
          x={20}
          y={padding.top + chartHeight / 2}
          textAnchor="middle"
          fontSize={12}
          fill="var(--text-secondary)"
          fontWeight={500}
          transform={`rotate(-90, 20, ${padding.top + chartHeight / 2})`}
        >
          Metric Value (%)
        </text>

        {/* Data lines */}
        {metrics.map(metric => {
          const path = getPath(metric.key);
          if (!path) return null;
          
          return (
            <g key={metric.key}>
              <path
                d={path}
                fill="none"
                stroke={metric.color}
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ 
                  filter: hoveredPoint?.metric === metric.label ? 'drop-shadow(0 0 4px ' + metric.color + ')' : 'none',
                  transition: 'filter 0.2s'
                }}
              />
              {/* Data points */}
              {normalizedData
                .filter(d => d[metric.key as keyof typeof d] !== undefined && d[metric.key as keyof typeof d] !== null)
                .map((d, i) => {
                  const value = d[metric.key as keyof typeof d] as number;
                  const x = getX(d.epoch);
                  const y = padding.top + chartHeight - (value * 100 * yScale);
                  const isHovered = hoveredPoint?.epoch === d.epoch && hoveredPoint?.metric === metric.label;
                  
                  return (
                    <circle
                      key={`${metric.key}-point-${i}`}
                      cx={x}
                      cy={y}
                      r={isHovered ? 5 : 3}
                      fill={metric.color}
                      stroke="var(--bg-primary)"
                      strokeWidth={2}
                      opacity={isHovered ? 1 : 0.6}
                      style={{ transition: 'all 0.2s' }}
                    />
                  );
                })}
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${padding.left + chartWidth + 20}, ${padding.top})`}>
          {metrics.map((metric, i) => (
            <g key={`legend-${metric.key}`} transform={`translate(0, ${i * 25})`}>
              <line
                x1={0}
                y1={0}
                x2={20}
                y2={0}
                stroke={metric.color}
                strokeWidth={2}
              />
              <text
                x={28}
                y={4}
                fontSize={11}
                fill="var(--text-secondary)"
              >
                {metric.label}
              </text>
            </g>
          ))}
        </g>

        {/* Hover tooltip */}
        {hoveredPoint && (
          <g>
            {/* Vertical line at epoch */}
            <line
              x1={hoveredPoint.x}
              y1={padding.top}
              x2={hoveredPoint.x}
              y2={padding.top + chartHeight}
              stroke="var(--text-muted)"
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.5}
            />
            {/* Tooltip background */}
            <rect
              x={hoveredPoint.x + 10}
              y={Math.max(hoveredPoint.y - 35, padding.top)}
              width={140}
              height={28}
              rx={6}
              fill="var(--bg-secondary)"
              stroke="var(--border-primary)"
              strokeWidth={1}
            />
            {/* Tooltip text */}
            <text
              x={hoveredPoint.x + 20}
              y={Math.max(hoveredPoint.y - 35, padding.top) + 18}
              fontSize={11}
              fill="var(--text-secondary)"
              fontWeight={500}
            >
              Epoch {hoveredPoint.epoch}: {(hoveredPoint.value * 100).toFixed(1)}%
            </text>
          </g>
        )}
        
        {/* Notice for final metrics only */}
        {hasOnlyFinalMetrics && (
          <text
            x={width / 2}
            y={height - 5}
            textAnchor="middle"
            fontSize={11}
            fill="var(--text-muted)"
            fontStyle="italic"
          >
            Showing final metrics only (training history not available)
          </text>
        )}
      </svg>
    </div>
  );
};

export default LineChart;
