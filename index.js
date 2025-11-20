const fs = require('fs');
const path = require('path');
const readline = require('readline');
const NN = require('./nn.js');

// Load config.json
let config = {};
try {
  config = JSON.parse(fs.readFileSync(path.join(__dirname, 'config.json'), 'utf8'));
  console.log('Loaded config.json');
} catch {
  console.log('No config.json found, using defaults');
  config = {
    trainingDataFile: 'trainingData.json',
    netFile: 'net.json',
    memoryFile: 'memory.json',
    layers: [
      { nodes: 2 },
      { nodes: 8, activation: 'relu' },
      { nodes: 1, activation: 'sigmoid' }
    ],
    learningRate: 0.05,
    iterations: 5000
  };
}

// Load training data
let trainingData = [];
try {
  trainingData = JSON.parse(fs.readFileSync(config.trainingDataFile, 'utf8'));
  console.log(`Training samples: ${trainingData.length}`);
} catch {
  console.log(`No ${config.trainingDataFile} found.`);
}

// Load network or create new
let net;
try {
  const saved = fs.readFileSync(config.netFile, 'utf8');
  net = NN.fromJSON(saved);
  console.log(`Loaded existing ${config.netFile}`);
} catch {
  console.log("Creating new network...");
  net = new NN({
    layers: config.layers,
    learningRate: config.learningRate
  });
}

// Load memory
try {
  const mem = fs.readFileSync(config.memoryFile, 'utf8');
  net.loadMemory(mem);
  console.log(`Loaded ${config.memoryFile} with ${net.memory.length} samples`);
} catch {
  console.log(`No ${config.memoryFile} found.`);
}

// Train network
console.log("Training...");
net.train(trainingData, { iterations: config.iterations });
console.log("Training complete.");

// Save network & memory
fs.writeFileSync(config.netFile, net.toJSON());
fs.writeFileSync(config.memoryFile, JSON.stringify(net.memory, null, 2));

// ===============================
// AFTER TRAINING → ASK FOR INPUT
// ===============================
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function askInput() {
  rl.question("Enter input (comma separated, e.g. 1,0): ", (answer) => {
    try {
      const arr = answer.split(',').map(n => Number(n.trim()));
      const rawOut = net.run(arr);

      // ROUND OUTPUT HERE (6 decimals)
      const out = rawOut.map(v => Number(v.toFixed(6)));

      console.log("Output:", out);
    } catch (e) {
      console.log("Error:", e.message);
    }
    askInput(); // allow infinite runs
  });
}

askInput();
