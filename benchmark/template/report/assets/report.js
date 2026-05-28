// BRISE Benchmark Report - Interactive Features

function showTab(id, btn) {
    document.querySelectorAll('.tab-content').forEach(function(e) {
        e.style.display = 'none';
    });
    document.getElementById(id).style.display = 'block';
    document.querySelectorAll('.tab-btn').forEach(function(b) {
        b.classList.remove('active');
    });
    btn.classList.add('active');
    setTimeout(function() {
        document.querySelectorAll('.js-plotly-plot').forEach(function(p) {
            if(p.offsetParent !== null) Plotly.Plots.resize(p);
        });
    }, 50);
}

// Enhanced Export Configuration System
const currentExportConfig = {
    containerId: null,
    plotDiv: null,
    fileName: '',
    width: 1200,
    height: 600,
    legendPosition: null
};

const legendPositions = {
    'top-left': {x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top'},
    'top-center': {x: 0.5, y: 0.98, xanchor: 'center', yanchor: 'top'},
    'top-right': {x: 0.98, y: 0.98, xanchor: 'right', yanchor: 'top'},
    'middle-left': {x: 0.02, y: 0.5, xanchor: 'left', yanchor: 'middle'},
    'middle-right': {x: 0.98, y: 0.5, xanchor: 'right', yanchor: 'middle'},
    'bottom-left': {x: 0.02, y: 0.02, xanchor: 'left', yanchor: 'bottom'},
    'bottom-center': {x: 0.5, y: 0.02, xanchor: 'center', yanchor: 'bottom'},
    'bottom-right': {x: 0.98, y: 0.02, xanchor: 'right', yanchor: 'bottom'},
    'outside-right': {x: 1.02, y: 1, xanchor: 'left', yanchor: 'top'},
    'outside-bottom': {x: 0.5, y: -0.2, xanchor: 'center', yanchor: 'top'}
};

function updateDimensions() {
    const width = parseFloat(document.getElementById('export-width').value);
    const height = parseFloat(document.getElementById('export-height').value);

    currentExportConfig.width = Math.round(width);
    currentExportConfig.height = Math.round(height);
}

function applyPreset() {
    const presetKey = document.getElementById('preset-select').value;
    if (presetKey === 'current') {
        document.getElementById('export-width').value = 1200;
        document.getElementById('export-height').value = 600;
        currentExportConfig.width = 1200;
        currentExportConfig.height = 600;
    }
}

function selectLegendPosition(position) {
    document.querySelectorAll('.legend-pos').forEach(function(el) {
        el.classList.remove('selected');
    });
    event.target.classList.add('selected');
    currentExportConfig.legendPosition = position;
}

function exportPlotAsSVG(containerId, fileName) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Container not found:', containerId);
        return;
    }
    const plotDiv = container.querySelector('.js-plotly-plot');
    if (!plotDiv) {
        console.error('Plotly plot not found in container:', containerId);
        return;
    }

    // Store current config
    currentExportConfig.containerId = containerId;
    currentExportConfig.plotDiv = plotDiv;
    currentExportConfig.fileName = fileName;
    currentExportConfig.width = 1200;
    currentExportConfig.height = 600;

    // Reset dialog
    document.getElementById('export-width').value = 1200;
    document.getElementById('export-height').value = 600;
    document.getElementById('preset-select').value = 'current';
    document.querySelectorAll('.legend-pos').forEach(function(el) {
        el.classList.remove('selected');
    });
    currentExportConfig.legendPosition = null;

    // Show dialog
    document.getElementById('export-dialog').classList.add('active');
}

function closeExportDialog() {
    document.getElementById('export-dialog').classList.remove('active');
}

function performExport() {
    const plotDiv = currentExportConfig.plotDiv;
    const filename = currentExportConfig.fileName;

    // Clone the plot configuration
    const exportLayout = JSON.parse(JSON.stringify(plotDiv.layout));

    // Apply legend position if selected
    if (currentExportConfig.legendPosition && currentExportConfig.legendPosition !== 'current') {
        var legendPos = legendPositions[currentExportConfig.legendPosition];
        exportLayout.legend = exportLayout.legend || {};
        Object.assign(exportLayout.legend, legendPos);

        // Add semi-transparent background for inside positions
        if (currentExportConfig.legendPosition !== 'outside') {
            exportLayout.legend.bgcolor = 'rgba(255,255,255,0.8)';
            exportLayout.legend.bordercolor = 'rgba(0,0,0,0.2)';
            exportLayout.legend.borderwidth = 1;
        }
    }

    // Update the plot temporarily for export
    Plotly.relayout(plotDiv, exportLayout).then(function() {
        // Export with custom dimensions
        Plotly.downloadImage(plotDiv, {
            format: 'svg',
            width: Math.round(currentExportConfig.width),
            height: Math.round(currentExportConfig.height),
            filename: filename
        }).then(function() {
            closeExportDialog();
        });
    });
}

window.addEventListener('load', function() {
    setTimeout(function() {
        document.querySelectorAll('.js-plotly-plot').forEach(function(p) {
            Plotly.Plots.resize(p);
        });
    }, 100);
});

