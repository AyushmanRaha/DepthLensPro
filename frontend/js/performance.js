"use strict";

// METRICS DASHBOARD
// ══════════════════════════════════════════════════════════════
function updateMetrics() {
  const s=state.session, lats=s.latencies, cache=state.cacheMetrics;
  el.metricTotal.textContent=s.total;
  el.metricCached.textContent=cache ? cache.totalHits : s.cached;
  if (cache) {
    const cacheCell = el.metricCached.closest(".metric-cell");
    if (cacheCell) {
      cacheCell.title = `Cache hits ${cache.totalHits} · misses ${cache.cacheMisses} · keys ${cache.keyspaceSize} · backend ${cache.backend} · TTL ${cache.ttlSeconds}s`;
    }
  }
  el.metricErrors.textContent=s.errors;
  el.metricTotalTime.textContent=`${(s.totalInferenceMs/1000).toFixed(1)} s`;
  if (lats.length) {
    const avg=lats.reduce((a,b)=>a+b,0)/lats.length;
    el.metricAvgLatency.textContent=avg.toFixed(0);
    el.metricMinLat.textContent=Math.min(...lats).toFixed(0);
    el.metricMaxLat.textContent=Math.max(...lats).toFixed(0);
    el.metricThroughput.textContent=(60000/avg).toFixed(1);
  }
}

// ══════════════════════════════════════════════════════════════
