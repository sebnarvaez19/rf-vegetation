var landsat_functions = require("users/statgis/package:landsat_functions.js");
var landsat_scaler = require("users/statgis/package:landsat_functions.js").landsat_scaler;

var lims = [-71.60608674813004, 11.829659967244739, -71.24341428099824, 12.100464634418024];
var bbox = ee.Geometry.BBox(lims[0], lims[1], lims[2], lims[3]);

var L7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
           .filterDate("2010-01-01", "2010-12-31")
           .filterBounds(bbox)
           .map(landsat_functions.landsat_cloud_mask)
           .map(landsat_scaler);

var vis = {bands: ["SR_B3", "SR_B2", "SR_B1"], min: 0.0, max: 0.3};

Map.addLayer(L7.mean(), vis);
// Map.addLayer(bbox);
Map.centerObject(bbox);