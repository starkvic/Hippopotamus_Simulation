// Global variables
let tableData;             
let iterationsData = [];   // Parsed CSV data; each element holds { iteration, power, voltage }
let currentIteration = 0;  
let totalIterations = 0;   // Number of iterations (rows) in the CSV
let numObjects = -1;       // Number of objects (set dynamically from CSV)
let objectTrails = [];     // Each object's history, stored as an array of { x, y } positions

// Global min/max ranges for scaling
// Voltage is taken from the Population array (x-axis)
let globalMinVoltage = Infinity;
let globalMaxVoltage = -Infinity;
// Power is taken from the Fitness Array (y-axis)
let globalMinPower = Infinity;
let globalMaxPower = -Infinity;

// Define margins for the plotting area
let leftMargin = 80;      // For y-axis labels
let bottomMargin = 80;    // For x-axis labels
let topMargin = 20;       // Small top margin for breathing room
let rightMargin = 0;      // No extra margin on the right

// Image variables
let hippoImg;         // Normal hippo image (for regular objects)
let hippoImgGreen;    // Highlighted hippo image (for best objects)
let hippoWidth = 60;  // Adjust icon width as desired
let hippoHeight = 60; // Adjust icon height as desired

// Preload the CSV file and images.
// The CSV file is expected to have headers "Iteration", "Fitness Array", and "Population".
// Here, "Fitness Array" provides power data (used for y-axis)
// and "Population" provides voltage data (used for x-axis).
function preload() {
  // Update the filename here as needed (e.g., "algorithm_name.csv")
  tableData = loadTable('/static/simulation_app/csv/abc_detailed_results.csv', 'csv', 'header', onCSVLoaded, onCSVError);
  hippoImg = loadImage('/static/simulation_app/images/hippo.png');
  hippoImgGreen = loadImage('/static/simulation_app/images/hippo1.png');
}

function onCSVLoaded() {
  console.log("CSV loaded successfully.");
  console.log("CSV columns:", tableData.columns);
}

function onCSVError(err) {
  console.error("Error loading CSV:", err);
}

function setup() {
  createCanvas(800, 600);
  textSize(12);

  // Process CSV data row by row.
  let numRows = tableData.getRowCount();
  for (let i = 0; i < numRows; i++) {
    let iterationVal = Number(tableData.getString(i, 'Iteration'));

    // Parse "Fitness Array" as power data (y-axis).
    let fitnessStr = tableData.getString(i, 'Fitness Array').replace("[", "").replace("]", "");
    let powerArray = fitnessStr.split(',').map(val => Number(val.trim()));

    // Parse "Population" as voltage data (x-axis).
    let populationStr = tableData.getString(i, 'Population Array').replace("[", "").replace("]", "");
    let voltageArray = populationStr.split(',').map(val => Number(val.trim()));

    // Ensure both arrays have the same length.
    if (powerArray.length !== voltageArray.length) {
      console.warn(`Row ${i} skipped: Mismatch in array lengths (power: ${powerArray.length}, voltage: ${voltageArray.length}).`);
      continue;
    }
    
    // If numObjects hasn't been set yet, use this row's count.
    if (numObjects < 0) {
      numObjects = powerArray.length;
      console.log("Number of objects set to:", numObjects);
    } else if (numObjects !== powerArray.length) {
      console.warn(`Row ${i} skipped: Expected ${numObjects} items, but got ${powerArray.length}.`);
      continue;
    }

    // Add row to iterationsData.
    iterationsData.push({
      iteration: iterationVal,
      power: powerArray,     // Power data from "Fitness Array" → y-axis
      voltage: voltageArray  // Voltage data from "Population" → x-axis
    });

    // Update global ranges.
    for (let v of voltageArray) {
      if (v < globalMinVoltage) globalMinVoltage = v;
      if (v > globalMaxVoltage) globalMaxVoltage = v;
    }
    for (let p of powerArray) {
      if (p < globalMinPower) globalMinPower = p;
      if (p > globalMaxPower) globalMaxPower = p;
    }
  }

  // Set totalIterations based on the number of parsed rows.
  totalIterations = iterationsData.length;
  if (totalIterations === 0) {
    console.error("No valid CSV rows found.");
    noLoop();
    return;
  }

  // Pad the ranges by 10% so points don't lie exactly on the borders.
  globalMaxVoltage *= 1.1;
  globalMinVoltage *= 0.9;
  globalMaxPower   *= 1.1;
  globalMinPower   *= 0.9;

  // Initialize objectTrails for each object.
  for (let i = 0; i < numObjects; i++) {
    objectTrails[i] = [];
  }

  // Set initial positions from the first valid row.
  let firstRow = iterationsData[0];
  for (let i = 0; i < numObjects; i++) {
    // x-axis: Voltage (from "Population")
    let xPos = map(firstRow.voltage[i], globalMinVoltage, globalMaxVoltage, leftMargin, width - rightMargin);
    xPos = constrain(xPos, leftMargin, width - rightMargin);
    // y-axis: Power (from "Fitness Array") with inverted mapping so higher power is at the top.
    let yPos = map(firstRow.power[i], globalMinPower, globalMaxPower, height - bottomMargin, topMargin);
    yPos = constrain(yPos, topMargin, height - bottomMargin);
    objectTrails[i].push({ x: xPos, y: yPos });
  }

  currentIteration = 0;
}

function draw() {
  background(220);
  drawAxes();

  let currentRow = iterationsData[currentIteration];

  // Identify best voltage (maximum voltage from the voltage array).
  let bestVoltIndex = 0;
  let bestVoltage = currentRow.voltage[0];
  for (let i = 1; i < numObjects; i++) {
    if (currentRow.voltage[i] > bestVoltage) {
      bestVoltage = currentRow.voltage[i];
      bestVoltIndex = i;
    }
  }

  // Identify best power (maximum power from the power array).
  let bestPowIndex = 0;
  let bestPower = currentRow.power[0];
  for (let i = 1; i < numObjects; i++) {
    if (currentRow.power[i] > bestPower) {
      bestPower = currentRow.power[i];
      bestPowIndex = i;
    }
  }

  // First, draw all normal objects using the normal hippo image.
  for (let i = 0; i < numObjects; i++) {
    if (i !== bestVoltIndex && i !== bestPowIndex) {
      let lastPoint = objectTrails[i][objectTrails[i].length - 1];
      image(hippoImg, lastPoint.x - hippoWidth / 2, lastPoint.y - hippoHeight / 2, hippoWidth, hippoHeight);
    }
  }

  // Then draw the best voltage object using the green hippo image.
  let bestVoltPos = objectTrails[bestVoltIndex][objectTrails[bestVoltIndex].length - 1];
  image(hippoImgGreen, bestVoltPos.x - hippoWidth / 2, bestVoltPos.y - hippoHeight / 2, hippoWidth, hippoHeight);

  // If the best power object is different, draw it in green as well.
  if (bestPowIndex !== bestVoltIndex) {
    let bestPowPos = objectTrails[bestPowIndex][objectTrails[bestPowIndex].length - 1];
    image(hippoImgGreen, bestPowPos.x - hippoWidth / 2, bestPowPos.y - hippoHeight / 2, hippoWidth, hippoHeight);
  }

  // Draw floating labels above the best objects.
  fill(0);
  noStroke();
  textAlign(CENTER, BOTTOM);
  text(`Best Voltage: ${nf(bestVoltage, 0, 2)}`, bestVoltPos.x, bestVoltPos.y - (hippoHeight / 2) - 5);
  let labelOffset = (bestVoltIndex === bestPowIndex) ? 25 : 10;
  let bestPowPosForLabel = objectTrails[bestPowIndex][objectTrails[bestPowIndex].length - 1];
  text(`Best Power: ${nf(bestPower, 0, 2)}`, bestPowPosForLabel.x, bestPowPosForLabel.y - (hippoHeight / 2) - labelOffset);

  // Update to the next iteration every 30 frames.
  if (frameCount % 30 === 0) {
    if (currentIteration < totalIterations - 1) {
      currentIteration++;
      let newRow = iterationsData[currentIteration];
      for (let i = 0; i < numObjects; i++) {
        let xPos = map(newRow.voltage[i], globalMinVoltage, globalMaxVoltage, leftMargin, width - rightMargin);
        xPos = constrain(xPos, leftMargin, width - rightMargin);
        let yPos = map(newRow.power[i], globalMinPower, globalMaxPower, height - bottomMargin, topMargin);
        yPos = constrain(yPos, topMargin, height - bottomMargin);
        objectTrails[i].push({ x: xPos, y: yPos });
      }
    }
  }

  // Display iteration info, e.g., "Iteration: 5 / 200"
  fill(0);
  textAlign(LEFT, TOP);
  text(`Iteration: ${iterationsData[currentIteration].iteration} / ${totalIterations}`, leftMargin, topMargin + 5);
}

function drawAxes() {
  stroke(0);
  strokeWeight(2);

  // Draw border rectangle for the plotting area.
  noFill();
  rect(leftMargin, topMargin, width - leftMargin - rightMargin, height - bottomMargin - topMargin);

  // Draw the x-axis (Voltage)
  line(leftMargin, height - bottomMargin, width - rightMargin, height - bottomMargin);
  let tickSpacing = 50;
  for (let x = leftMargin; x <= width - rightMargin; x += tickSpacing) {
    line(x, height - bottomMargin - 5, x, height - bottomMargin + 5);
    noStroke();
    fill(0);
    textAlign(CENTER, TOP);
    let voltLabel = map(x, leftMargin, width - rightMargin, globalMinVoltage, globalMaxVoltage);
    text(nf(voltLabel, 0, 2), x, height - bottomMargin + 8);
    stroke(0);
  }
  // Explicit label for maximum voltage at the right edge.
  noStroke();
  fill(0);
  textAlign(CENTER, TOP);
  text(nf(globalMaxVoltage, 0, 2), width - rightMargin, height - bottomMargin + 8);
  stroke(0);

  // X-axis label: "Voltage"
  noStroke();
  fill(0);
  textAlign(CENTER, CENTER);
  text("Voltage", (leftMargin + (width - rightMargin)) / 2, height - bottomMargin / 2);

  // Draw the y-axis (Power)
  line(leftMargin, topMargin, leftMargin, height - bottomMargin);
  for (let y = height - bottomMargin; y >= topMargin; y -= tickSpacing) {
    line(leftMargin - 5, y, leftMargin + 5, y);
    noStroke();
    fill(0);
    textAlign(RIGHT, CENTER);
    let powerLabel = map(y, height - bottomMargin, topMargin, globalMinPower, globalMaxPower);
    text(nf(powerLabel, 0, 2), leftMargin - 8, y);
    stroke(0);
  }
  // Explicit label for maximum power at the top.
  noStroke();
  fill(0);
  textAlign(RIGHT, CENTER);
  text(nf(globalMaxPower, 0, 2), leftMargin - 8, topMargin);
  stroke(0);

  // Y-axis label: "Power"
  push();
  translate(leftMargin / 2, (topMargin + (height - bottomMargin)) / 2);
  rotate(-HALF_PI);
  noStroke();
  fill(0);
  text("Power", 0, 0);
  pop();
}
