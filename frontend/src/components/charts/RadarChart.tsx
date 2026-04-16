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
  '#22c55e',  // Green - bright
  '#4ade80',  // Light green
  '#fbbf24',  // Amber/Yellow
  '#a855f7'   // Purple
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

  const size = 400;
  const center = size / 2;
  const radius = 140;
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
                stroke="#44403c"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                opacity={0.8}
              />
              <text
                x={center + 3}
                y={center - radius * level}
                fontSize={9}
                fill="#a8a29e"
                fontWeight={500}
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
            const labelX = center + (radius + 28) * Math.cos(angle);
            const labelY = center + (radius + 28) * Math.sin(angle);
            
            return (
              <g key={metric.key}>
                <line
                  x1={center}
                  y1={center}
                  x2={endX}
                  y2={endY}
                  stroke="#57534e"
                  strokeWidth={1.5}
                  opacity={0.6}
                />
                <text
                  x={labelX}
                  y={labelY}
                  fontSize={11}
                  fontWeight={600}
                  fill="#e7e5e4"
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
            r={4}
            fill="#a8a29e"
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
                  fillOpacity={0.35}
                  stroke="none"
                />
                {/* Border */}
                <path
                  d={path}
                  fill="none"
                  stroke={color}
                  strokeWidth={3}
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
                      r={isHovered ? 10 : 7}
                      fill={color}
                      stroke="#0c0a09"
                      strokeWidth={2}
                      opacity={isHovered ? 1 : 0.9}
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
                x={hoveredPoint.x + 8}
                y={hoveredPoint.y - 20}
                width={100}
                height={22}
                rx={4}
                fill="#1c1917"
                stroke="#44403c"
                strokeWidth={1}
              />
              <text
                x={hoveredPoint.x + 58}
                y={hoveredPoint.y - 4}
                fontSize={9}
                fill="#a8a29e"
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
