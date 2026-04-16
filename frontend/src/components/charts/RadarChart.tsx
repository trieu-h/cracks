import React, { useState } from 'react';

interface Metric {
  key: string;
  label: string;
  max: number;
}

interface Session {
  id: string;
  latest_metrics?: {
    [key: string]: number;
  };
}

interface RadarChartProps {
  sessions: Session[];
  metrics: Metric[];
}

const COLORS = [
  'var(--accent-primary)',
  'var(--success-text)', 
  'var(--warning-text)',
  '#8b5cf6'
];

const RadarChart: React.FC<RadarChartProps> = ({ sessions, metrics }) => {
  const [hiddenModels, setHiddenModels] = useState<Set<number>>(new Set());
  const [hoveredPoint, setHoveredPoint] = useState<{
    sessionIndex: number;
    metricIndex: number;
    value: number;
    x: number;
    y: number;
  } | null>(null);

  const size = 500;
  const center = size / 2;
  const radius = 180;
  const angleStep = (2 * Math.PI) / metrics.length;

  // Calculate normalized values (0-1)
  const getNormalizedValue = (session: Session, metric: Metric) => {
    const value = session.latest_metrics?.[metric.key];
    if (value === undefined || value === null) return 0;
    return Math.min(Math.max(value / metric.max, 0), 1);
  };

  // Get point coordinates for a metric at a normalized distance
  const getPoint = (metricIndex: number, normalizedValue: number) => {
    const angle = metricIndex * angleStep - Math.PI / 2;
    const r = normalizedValue * radius;
    return {
      x: center + r * Math.cos(angle),
      y: center + r * Math.sin(angle)
    };
  };

  // Generate polygon path for a session
  const getPolygonPath = (session: Session) => {
    const points = metrics.map((metric, i) => {
      const value = getNormalizedValue(session, metric);
      return getPoint(i, value);
    });
    
    return points.reduce((path, point, i) => {
      if (i === 0) return `M ${point.x} ${point.y}`;
      return `${path} L ${point.x} ${point.y}`;
    }, '') + ' Z';
  };

  // Toggle model visibility
  const toggleModel = (index: number) => {
    setHiddenModels(prev => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  // Handle mouse move for hover
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Find nearest metric vertex
    let minDistance = Infinity;
    let nearestPoint: typeof hoveredPoint = null;
    
    sessions.forEach((session, sessionIndex) => {
      if (hiddenModels.has(sessionIndex)) return;
      
      metrics.forEach((metric, metricIndex) => {
        const value = getNormalizedValue(session, metric);
        const point = getPoint(metricIndex, value);
        const distance = Math.sqrt(
          Math.pow(mouseX - point.x, 2) + Math.pow(mouseY - point.y, 2)
        );
        
        if (distance < 30 && distance < minDistance) {
          minDistance = distance;
          nearestPoint = {
            sessionIndex,
            metricIndex,
            value: session.latest_metrics?.[metric.key] || 0,
            x: point.x,
            y: point.y
          };
        }
      });
    });
    
    setHoveredPoint(nearestPoint);
  };

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Legend */}
      <div className="flex items-center gap-4 flex-wrap justify-center">
        {sessions.map((session, index) => {
          const isHidden = hiddenModels.has(index);
          const color = COLORS[index % COLORS.length];
          
          return (
            <button
              key={session.id}
              onClick={() => toggleModel(index)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg transition-all"
              style={{
                background: isHidden ? 'var(--bg-tertiary)' : 'var(--bg-secondary)',
                border: isHidden ? '1px solid var(--border-secondary)' : `2px solid ${color}`,
                opacity: isHidden ? 0.5 : 1
              }}
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{ background: color }}
              />
              <span className="text-sm font-medium" style={{ color: isHidden ? 'var(--text-muted)' : 'var(--text-secondary)' }}>
                Run {session.id}
              </span>
            </button>
          );
        })}
      </div>

      {/* Radar Chart */}
      <div className="relative">
        <svg
          width={size}
          height={size}
          className="cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredPoint(null)}
        >
          {/* Grid circles */}
          {[0.25, 0.5, 0.75, 1].map(level => (
            <g key={level}>
              <circle
                cx={center}
                cy={center}
                r={radius * level}
                fill="none"
                stroke="var(--border-primary)"
                strokeWidth={1}
                strokeDasharray="4 4"
                opacity={0.5}
              />
              <text
                x={center + 5}
                y={center - radius * level}
                fontSize={10}
                fill="var(--text-muted)"
                textAnchor="start"
              >
                {Math.round(level * 100)}%
              </text>
            </g>
          ))}

          {/* Axis lines and labels */}
          {metrics.map((metric, i) => {
            const angle = i * angleStep - Math.PI / 2;
            const endX = center + radius * Math.cos(angle);
            const endY = center + radius * Math.sin(angle);
            const labelX = center + (radius + 30) * Math.cos(angle);
            const labelY = center + (radius + 30) * Math.sin(angle);
            
            return (
              <g key={metric.key}>
                <line
                  x1={center}
                  y1={center}
                  x2={endX}
                  y2={endY}
                  stroke="var(--border-primary)"
                  strokeWidth={1}
                  opacity={0.3}
                />
                <text
                  x={labelX}
                  y={labelY}
                  fontSize={12}
                  fontWeight={500}
                  fill="var(--text-secondary)"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {metric.label}
                </text>
              </g>
            );
          })}

          {/* Center point */}
          <circle
            cx={center}
            cy={center}
            r={3}
            fill="var(--text-muted)"
          />

          {/* Model polygons */}
          {sessions.map((session, index) => {
            if (hiddenModels.has(index)) return null;
            
            const color = COLORS[index % COLORS.length];
            const path = getPolygonPath(session);
            
            return (
              <g key={session.id}>
                {/* Fill */}
                <path
                  d={path}
                  fill={color}
                  fillOpacity={0.2}
                  stroke="none"
                />
                {/* Border */}
                <path
                  d={path}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  strokeLinejoin="round"
                />
                {/* Data points */}
                {metrics.map((metric, i) => {
                  const value = getNormalizedValue(session, metric);
                  const point = getPoint(i, value);
                  const isHovered = hoveredPoint?.sessionIndex === index && hoveredPoint?.metricIndex === i;
                  
                  return (
                    <circle
                      key={`${session.id}-${i}`}
                      cx={point.x}
                      cy={point.y}
                      r={isHovered ? 8 : 5}
                      fill={color}
                      stroke="var(--bg-primary)"
                      strokeWidth={2}
                      opacity={isHovered ? 1 : 0.8}
                      style={{ transition: 'all 0.2s ease' }}
                    />
                  );
                })}
              </g>
            );
          })}

          {/* Hover tooltip */}
          {hoveredPoint && (
            <g>
              <rect
                x={hoveredPoint.x + 10}
                y={hoveredPoint.y - 40}
                width={120}
                height={35}
                rx={6}
                fill="var(--bg-secondary)"
                stroke="var(--border-primary)"
                strokeWidth={1}
              />
              <text
                x={hoveredPoint.x + 70}
                y={hoveredPoint.y - 22}
                fontSize={11}
                fontWeight={600}
                fill="var(--text-secondary)"
                textAnchor="middle"
              >
                Run {sessions[hoveredPoint.sessionIndex].id}
              </text>
              <text
                x={hoveredPoint.x + 70}
                y={hoveredPoint.y - 8}
                fontSize={10}
                fill="var(--text-muted)"
                textAnchor="middle"
              >
                {metrics[hoveredPoint.metricIndex].label}: {(hoveredPoint.value * 100).toFixed(1)}%
              </text>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
};

export default RadarChart;
