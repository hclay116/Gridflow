import React, { useEffect, useState, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const MAX_VEHICLES = 100;

const TrafficSimulator = () => {
  const [settings, setSettings] = useState({
    carDensity: 20, // starting with more vehicles
    trafficLightTiming: 20,
    isAutoMode: true,
    carSpeed: 0.15,
    timeOfDay: 'day',
  });
  // activeLight now holds a configuration string: either "NS-green" or "EW-green"
  const [activeLight, setActiveLight] = useState('NS-green');

  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const vehiclesRef = useRef([]);
  const trafficLightsRef = useRef([]);
  // We'll still use lightStateRef for the baseline timer; however, when in RL mode,
  // the state from the server will be used.
  const lightStateRef = useRef('NS-green');
  const frameIdRef = useRef(null);
  const timerRef = useRef(null);
  const spawnIntervalRef = useRef(null);

  // --- Capture & send canvas image to RL endpoint periodically ---
  useEffect(() => {
    // Only if automatic mode is enabled.
    if (!settings.isAutoMode) return;
       const rlInterval = setInterval(() => {
        if (rendererRef.current) {
          const canvas = rendererRef.current.domElement;
          const dataURL = canvas.toDataURL("image/jpeg", 0.5);
          const base64 = dataURL.split(",")[1];
          fetch("http://localhost:5001/policy/rl", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: base64 })
          })
            .then((response) => response.json())
            .then((data) => {
              if (data.state === "NS-green") {
                setActiveLight("NS-green");
                lightStateRef.current = "NS-green";
              } else {
                setActiveLight("EW-green");
                lightStateRef.current = "EW-green";
              }
            })
            .catch((err) => console.error("Error in RL policy fetch:", err));
        }
      }, 1000); // every 1 second

    return () => clearInterval(rlInterval);
  }, [settings.isAutoMode]);

  useEffect(() => {
    // --- Initialization ---
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    updateEnvironment(settings.timeOfDay);

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(30, 25, 30);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 10;
    controls.maxDistance = 60;
    controls.maxPolarAngle = Math.PI / 2 - 0.1;
    controlsRef.current = controls;

    createCityEnvironment();

    // --- Resize Handler ---
    const handleResize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // --- Animation Loop (24/7) ---
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      updateSimulation();
      controlsRef.current?.update();
      renderer.render(scene, camera);
    };
    animate();

    // --- Traffic Light Timer (Baseline or fallback) ---
    startTrafficLightTimer();

    // --- Initial Vehicle Spawning ---
    resetSimulation();

    // --- Continuous Vehicle Spawner ---
    spawnIntervalRef.current = setInterval(() => {
      if (sceneRef.current && vehiclesRef.current.length < MAX_VEHICLES) {
        const vehicle = createVehicle();
        sceneRef.current.add(vehicle);
        vehiclesRef.current.push(vehicle);
      }
    }, 1000);

    // --- Cleanup ---
    return () => {
      window.removeEventListener('resize', handleResize);
      if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
      if (spawnIntervalRef.current) clearInterval(spawnIntervalRef.current);
      if (controlsRef.current) controlsRef.current.dispose();
      if (rendererRef.current) {
        rendererRef.current.dispose();
        if (mountRef.current && rendererRef.current.domElement) {
          mountRef.current.removeChild(rendererRef.current.domElement);
        }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Environment Setup ---
  const updateEnvironment = (timeOfDay) => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;
    scene.children = scene.children.filter(child =>
      !(child instanceof THREE.AmbientLight ||
        child instanceof THREE.DirectionalLight ||
        child instanceof THREE.HemisphereLight)
    );
    if (timeOfDay === 'day') {
      scene.background = new THREE.Color(0x87CEEB);
      const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
      scene.add(hemi);
      const dir = new THREE.DirectionalLight(0xffffff, 1);
      dir.position.set(50, 100, 50);
      dir.castShadow = true;
      dir.shadow.mapSize.width = 2048;
      dir.shadow.mapSize.height = 2048;
      dir.shadow.camera.near = 10;
      dir.shadow.camera.far = 200;
      dir.shadow.camera.left = -50;
      dir.shadow.camera.right = 50;
      dir.shadow.camera.top = 50;
      dir.shadow.camera.bottom = -50;
      scene.add(dir);
    } else if (timeOfDay === 'sunset') {
      scene.background = new THREE.Color(0xFFA07A);
      const hemi = new THREE.HemisphereLight(0xffdcb4, 0x422414, 0.7);
      scene.add(hemi);
      const dir = new THREE.DirectionalLight(0xfc9e5d, 0.8);
      dir.position.set(-50, 20, 0);
      dir.castShadow = true;
      dir.shadow.mapSize.width = 2048;
      dir.shadow.mapSize.height = 2048;
      scene.add(dir);
    } else if (timeOfDay === 'night') {
      scene.background = new THREE.Color(0x0a1a2a);
      const ambient = new THREE.AmbientLight(0x222266, 0.3);
      scene.add(ambient);
    }
  };

  const createCityEnvironment = () => {
    const scene = sceneRef.current;
    const groundGeo = new THREE.PlaneGeometry(100, 100);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.8 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
    createIntersection();
    createBuildings();
    createTrafficLights();
    createStreetLamps();
    createGreenery();
  };

  // --- Intersection, Roads, Markings, and Sidewalks ---
  const createIntersection = () => {
    const scene = sceneRef.current;
    const roadMat = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.7 });
    const nsRoadGeo = new THREE.PlaneGeometry(12, 100);
    const nsRoad = new THREE.Mesh(nsRoadGeo, roadMat);
    nsRoad.rotation.x = -Math.PI / 2;
    nsRoad.position.y = 0.01;
    nsRoad.receiveShadow = true;
    scene.add(nsRoad);
    const ewRoadGeo = new THREE.PlaneGeometry(100, 12);
    const ewRoad = new THREE.Mesh(ewRoadGeo, roadMat);
    ewRoad.rotation.x = -Math.PI / 2;
    ewRoad.position.y = 0.01;
    ewRoad.receiveShadow = true;
    scene.add(ewRoad);
    createRoadMarkings();
    createSidewalks();
  };

  const createRoadMarkings = () => {
    const scene = sceneRef.current;
    const markMat = new THREE.MeshStandardMaterial({ color: 0xFFFFFF, roughness: 0.5 });
    for (let i = -48; i <= 48; i += 8) {
      if (Math.abs(i) > 8) {
        const nsDashGeo = new THREE.PlaneGeometry(0.5, 4);
        const nsDash = new THREE.Mesh(nsDashGeo, markMat);
        nsDash.rotation.x = -Math.PI / 2;
        nsDash.position.set(0, 0.02, i);
        nsDash.receiveShadow = true;
        scene.add(nsDash);
      }
      if (Math.abs(i) > 8) {
        const ewDashGeo = new THREE.PlaneGeometry(4, 0.5);
        const ewDash = new THREE.Mesh(ewDashGeo, markMat);
        ewDash.rotation.x = -Math.PI / 2;
        ewDash.position.set(i, 0.02, 0);
        ewDash.receiveShadow = true;
        scene.add(ewDash);
      }
    }
    for (let i = -5; i <= 5; i += 2) {
      const northCWGeo = new THREE.PlaneGeometry(8, 0.8);
      const northCW = new THREE.Mesh(northCWGeo, markMat);
      northCW.rotation.x = -Math.PI / 2;
      northCW.position.set(i, 0.02, 8);
      northCW.receiveShadow = true;
      scene.add(northCW);
      const southCWGeo = new THREE.PlaneGeometry(8, 0.8);
      const southCW = new THREE.Mesh(southCWGeo, markMat);
      southCW.rotation.x = -Math.PI / 2;
      southCW.position.set(i, 0.02, -8);
      southCW.receiveShadow = true;
      scene.add(southCW);
      const eastCWGeo = new THREE.PlaneGeometry(0.8, 8);
      const eastCW = new THREE.Mesh(eastCWGeo, markMat);
      eastCW.rotation.x = -Math.PI / 2;
      eastCW.position.set(8, 0.02, i);
      eastCW.receiveShadow = true;
      scene.add(eastCW);
      const westCWGeo = new THREE.PlaneGeometry(0.8, 8);
      const westCW = new THREE.Mesh(westCWGeo, markMat);
      westCW.rotation.x = -Math.PI / 2;
      westCW.position.set(-8, 0.02, i);
      westCW.receiveShadow = true;
      scene.add(westCW);
    }
    const northStopGeo = new THREE.PlaneGeometry(6, 0.8);
    const northStop = new THREE.Mesh(northStopGeo, markMat);
    northStop.rotation.x = -Math.PI / 2;
    northStop.position.set(3, 0.02, 6);
    northStop.receiveShadow = true;
    scene.add(northStop);
    const southStopGeo = new THREE.PlaneGeometry(6, 0.8);
    const southStop = new THREE.Mesh(southStopGeo, markMat);
    southStop.rotation.x = -Math.PI / 2;
    southStop.position.set(-3, 0.02, -6);
    southStop.receiveShadow = true;
    scene.add(southStop);
    const eastStopGeo = new THREE.PlaneGeometry(0.8, 6);
    const eastStop = new THREE.Mesh(eastStopGeo, markMat);
    eastStop.rotation.x = -Math.PI / 2;
    eastStop.position.set(6, 0.02, -3);
    eastStop.receiveShadow = true;
    scene.add(eastStop);
    const westStopGeo = new THREE.PlaneGeometry(0.8, 6);
    const westStop = new THREE.Mesh(westStopGeo, markMat);
    westStop.rotation.x = -Math.PI / 2;
    westStop.position.set(-6, 0.02, 3);
    westStop.receiveShadow = true;
    scene.add(westStop);
  };

  const createSidewalks = () => {
    const scene = sceneRef.current;
    const sideMat = new THREE.MeshStandardMaterial({ color: 0x999999, roughness: 0.6 });
    const nsEastSWGeo = new THREE.BoxGeometry(4, 0.3, 100);
    const nsEastSW = new THREE.Mesh(nsEastSWGeo, sideMat);
    nsEastSW.position.set(8, 0.15, 0);
    nsEastSW.receiveShadow = true;
    nsEastSW.castShadow = true;
    scene.add(nsEastSW);
    const nsWestSWGeo = new THREE.BoxGeometry(4, 0.3, 100);
    const nsWestSW = new THREE.Mesh(nsWestSWGeo, sideMat);
    nsWestSW.position.set(-8, 0.15, 0);
    nsWestSW.receiveShadow = true;
    nsWestSW.castShadow = true;
    scene.add(nsWestSW);
    const ewNorthSWGeo = new THREE.BoxGeometry(100, 0.3, 4);
    const ewNorthSW = new THREE.Mesh(ewNorthSWGeo, sideMat);
    ewNorthSW.position.set(0, 0.15, 8);
    ewNorthSW.receiveShadow = true;
    ewNorthSW.castShadow = true;
    scene.add(ewNorthSW);
    const ewSouthSWGeo = new THREE.BoxGeometry(100, 0.3, 4);
    const ewSouthSW = new THREE.Mesh(ewSouthSWGeo, sideMat);
    ewSouthSW.position.set(0, 0.15, -8);
    ewSouthSW.receiveShadow = true;
    ewSouthSW.castShadow = true;
    scene.add(ewSouthSW);
  };

  const createBuildings = () => {
    const scene = sceneRef.current;
    const positions = [
      { x: 20, z: 20 }, { x: 32, z: 18 }, { x: 25, z: 30 },
      { x: -20, z: 20 }, { x: -30, z: 25 }, { x: -22, z: 35 },
      { x: 20, z: -20 }, { x: 28, z: -28 }, { x: 35, z: -22 },
      { x: -20, z: -20 }, { x: -25, z: -30 }, { x: -35, z: -25 }
    ];
    positions.forEach(pos => createBuilding(pos.x, pos.z));
  };

  const createBuilding = (x, z) => {
    const scene = sceneRef.current;
    const height = 5 + Math.random() * 15;
    const width = 6 + Math.random() * 8;
    const depth = 6 + Math.random() * 8;
    const geometry = new THREE.BoxGeometry(width, height, depth);
    let material;
    const type = Math.floor(Math.random() * 3);
    if (type === 0) {
      material = new THREE.MeshStandardMaterial({ color: 0x7790a0, roughness: 0.2, metalness: 0.8 });
    } else if (type === 1) {
      material = new THREE.MeshStandardMaterial({ color: 0xa85032, roughness: 0.7 });
    } else {
      material = new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.5 });
    }
    const building = new THREE.Mesh(geometry, material);
    building.position.set(x, height / 2, z);
    building.castShadow = true;
    building.receiveShadow = true;
    scene.add(building);
    if (type !== 0) addBuildingWindows(building, width, height, depth);
    if (Math.random() > 0.5) addRooftopStructures(x, height, z, width, depth);
  };

  const addBuildingWindows = (building, width, height, depth) => {
    const scene = sceneRef.current;
    const winMat = new THREE.MeshStandardMaterial({
      color: 0xffffcc,
      emissive: settings.timeOfDay === 'night' ? 0xffffaa : 0x000000,
      emissiveIntensity: settings.timeOfDay === 'night' ? 0.5 : 0,
      roughness: 0.2,
    });
    const winSize = 0.8;
    const winDepth = 0.1;
    const rows = Math.floor(height / 2) - 1;
    const cols = Math.floor(width / 2) - 1;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const frontGeo = new THREE.BoxGeometry(winSize, winSize, winDepth);
        const frontWin = new THREE.Mesh(frontGeo, winMat);
        frontWin.position.set(
          building.position.x - width / 2 + 1 + col * 2,
          1 + row * 2,
          building.position.z + depth / 2 + 0.1
        );
        scene.add(frontWin);
        const backGeo = new THREE.BoxGeometry(winSize, winSize, winDepth);
        const backWin = new THREE.Mesh(backGeo, winMat);
        backWin.position.set(
          building.position.x - width / 2 + 1 + col * 2,
          1 + row * 2,
          building.position.z - depth / 2 - 0.1
        );
        backWin.rotation.y = Math.PI;
        scene.add(backWin);
      }
    }
    const sideCols = Math.floor(depth / 2) - 1;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < sideCols; col++) {
        const leftGeo = new THREE.BoxGeometry(winDepth, winSize, winSize);
        const leftWin = new THREE.Mesh(leftGeo, winMat);
        leftWin.position.set(
          building.position.x - width / 2 - 0.1,
          1 + row * 2,
          building.position.z - depth / 2 + 1 + col * 2
        );
        scene.add(leftWin);
        const rightGeo = new THREE.BoxGeometry(winDepth, winSize, winSize);
        const rightWin = new THREE.Mesh(rightGeo, winMat);
        rightWin.position.set(
          building.position.x + width / 2 + 0.1,
          1 + row * 2,
          building.position.z - depth / 2 + 1 + col * 2
        );
        scene.add(rightWin);
      }
    }
  };

  const addRooftopStructures = (x, height, z, width, depth) => {
    const scene = sceneRef.current;
    if (Math.random() > 0.5) {
      const baseGeo = new THREE.CylinderGeometry(0.8, 0.8, 1, 8);
      const baseMat = new THREE.MeshStandardMaterial({ color: 0x777777 });
      const base = new THREE.Mesh(baseGeo, baseMat);
      base.position.set(x + width / 4, height + 0.5, z - depth / 4);
      base.castShadow = true;
      scene.add(base);
      const towerGeo = new THREE.CylinderGeometry(1.2, 1.2, 1.8, 8);
      const towerMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
      const tower = new THREE.Mesh(towerGeo, towerMat);
      tower.position.set(x + width / 4, height + 1.9, z - depth / 4);
      tower.castShadow = true;
      scene.add(tower);
    } else {
      const acGeo = new THREE.BoxGeometry(2, 1.2, 2);
      const acMat = new THREE.MeshStandardMaterial({ color: 0x777777 });
      const ac = new THREE.Mesh(acGeo, acMat);
      ac.position.set(x - width / 4, height + 0.6, z + depth / 4);
      ac.castShadow = true;
      scene.add(ac);
      const ventGeo = new THREE.BoxGeometry(1.6, 0.1, 1.6);
      const ventMat = new THREE.MeshStandardMaterial({ color: 0x333333 });
      const vent = new THREE.Mesh(ventGeo, ventMat);
      vent.position.set(x - width / 4, height + 1.2, z + depth / 4);
      scene.add(vent);
    }
  };

  const createTrafficLights = () => {
    const scene = sceneRef.current;
    trafficLightsRef.current = [];
    const positions = [
      { x: 6, z: 6, rotationY: 0 },
      { x: -6, z: 6, rotationY: Math.PI / 2 },
      { x: -6, z: -6, rotationY: Math.PI },
      { x: 6, z: -6, rotationY: -Math.PI / 2 }
    ];
    positions.forEach(pos => {
      const light = createTrafficLight();
      light.position.set(pos.x, 0, pos.z);
      light.rotation.y = pos.rotationY;
      scene.add(light);
      trafficLightsRef.current.push(light);
    });
  };

  const createTrafficLight = () => {
    const group = new THREE.Group();
  
    // Post
    const postGeo = new THREE.CylinderGeometry(0.3, 0.3, 7, 8);
    const postMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
    const post = new THREE.Mesh(postGeo, postMat);
    post.position.y = 3.5;
    post.castShadow = true;
    group.add(post);
  
    // Arm - made wider
    const armGeo = new THREE.BoxGeometry(4, 0.5, 0.5);
    const armMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.position.set(-2, 6.8, 0);
    arm.castShadow = true;
    group.add(arm);
  
    // Housing - MUCH larger and more visible
    const housingGeo = new THREE.BoxGeometry(2, 5, 2);
    const housingMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
    const housing = new THREE.Mesh(housingGeo, housingMat);
    housing.position.set(-4, 6.3, 0);
    housing.castShadow = true;
    group.add(housing);
  
    // Adjusted light sizes: smaller but visible
    // Red light (top)
    const redGeo = new THREE.CircleGeometry(0.4, 32);
    const redMat = new THREE.MeshStandardMaterial({ 
      color: 0xff0000, 
      emissive: 0xff0000, 
      emissiveIntensity: 5.0 
    });
    const redLight = new THREE.Mesh(redGeo, redMat);
    // Position flush with the housing’s left face (left edge is x = -5) 
    // but inset slightly for a centered look.
    redLight.position.set(-4.9, 7.8, 0);
    redLight.rotation.y = Math.PI / 2;
    redLight.name = "redLight";
    group.add(redLight);
  
    // Yellow light (middle)
    const yellowGeo = new THREE.CircleGeometry(0.4, 32);
    const yellowMat = new THREE.MeshStandardMaterial({ 
      color: 0xffff00, 
      emissive: 0xffff00, 
      emissiveIntensity: 4.0
    });
    const yellowLight = new THREE.Mesh(yellowGeo, yellowMat);
    yellowLight.position.set(-4.9, 6.3, 0);
    yellowLight.rotation.y = Math.PI / 2;
    yellowLight.name = "yellowLight";
    group.add(yellowLight);
  
    // Green light (bottom)
    const greenGeo = new THREE.CircleGeometry(0.4, 32);
    const greenMat = new THREE.MeshStandardMaterial({ 
      color: 0x00ff00, 
      emissive: 0x00ff00, 
      emissiveIntensity: 5.0 
    });
    const greenLight = new THREE.Mesh(greenGeo, greenMat);
    greenLight.position.set(-4.9, 4.8, 0);
    greenLight.rotation.y = Math.PI / 2;
    greenLight.name = "greenLight";
    group.add(greenLight);
  
    // Adjust halos proportionally to the new light sizes
    const redHaloGeo = new THREE.RingGeometry(0.4, 1.0, 32);
    const redHaloMat = new THREE.MeshBasicMaterial({ 
      color: 0xff0000,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide
    });
    const redHalo = new THREE.Mesh(redHaloGeo, redHaloMat);
    redHalo.position.set(-4.9, 7.8, 0);
    redHalo.rotation.y = Math.PI / 2;
    redHalo.name = "redHalo";
    group.add(redHalo);
  
    const greenHaloGeo = new THREE.RingGeometry(0.4, 1.0, 32);
    const greenHaloMat = new THREE.MeshBasicMaterial({ 
      color: 0x00ff00,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide
    });
    const greenHalo = new THREE.Mesh(greenHaloGeo, greenHaloMat);
    greenHalo.position.set(-4.9, 4.8, 0);
    greenHalo.rotation.y = Math.PI / 2;
    greenHalo.name = "greenHalo";
    group.add(greenHalo);
  
    // Adjusted spotlights (positions aligned with the new light positions)
    const redPointLight = new THREE.SpotLight(0xff0000, 10, 30, Math.PI / 4, 0.5, 1);
    redPointLight.position.set(-4.9, 7.8, 0);
    redPointLight.target.position.set(-9, 0, 0); // Adjust target as needed
    redPointLight.name = "redPointLight";
    group.add(redPointLight);
    group.add(redPointLight.target);
  
    const greenPointLight = new THREE.SpotLight(0x00ff00, 10, 30, Math.PI / 4, 0.5, 1);
    greenPointLight.position.set(-4.9, 4.8, 0);
    greenPointLight.target.position.set(-9, 0, 0); // Adjust target as needed
    greenPointLight.name = "greenPointLight";
    group.add(greenPointLight);
    group.add(greenPointLight.target);
  
    return group;
  };
  
  

  const createStreetLamps = () => {
    const scene = sceneRef.current;
    const positions = [
      { x: 10, z: 15 }, { x: 10, z: -15 },
      { x: -10, z: 15 }, { x: -10, z: -15 },
      { x: 15, z: 10 }, { x: -15, z: 10 },
      { x: 15, z: -10 }, { x: -15, z: -10 }
    ];
    positions.forEach(pos => {
      const lamp = createStreetLamp();
      lamp.position.set(pos.x, 0, pos.z);
      scene.add(lamp);
    });
  };

  const createStreetLamp = () => {
    const group = new THREE.Group();
    const postGeo = new THREE.CylinderGeometry(0.15, 0.15, 5, 8);
    const postMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
    const post = new THREE.Mesh(postGeo, postMat);
    post.position.y = 2.5;
    post.castShadow = true;
    group.add(post);
    const headGeo = new THREE.CylinderGeometry(0.5, 0.7, 0.8, 8);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x333333 });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 5.3;
    head.castShadow = true;
    group.add(head);
    const lightGeo = new THREE.CircleGeometry(0.4, 16);
    const lightMat = new THREE.MeshStandardMaterial({
      color: 0xffffee,
      emissive: 0xffffee,
      emissiveIntensity: settings.timeOfDay === 'night' ? 1 : 0.2
    });
    const light = new THREE.Mesh(lightGeo, lightMat);
    light.position.set(0, 5, 0);
    light.rotation.x = -Math.PI / 2;
    group.add(light);
    if (settings.timeOfDay === 'night') {
      const pointLight = new THREE.PointLight(0xffffee, 1, 20);
      pointLight.position.set(0, 5, 0);
      group.add(pointLight);
    }
    return group;
  };

  const createGreenery = () => {
    const scene = sceneRef.current;
    const positions = [
      { x: 15, z: 15 }, { x: 15, z: -15 },
      { x: -15, z: 15 }, { x: -15, z: -15 },
      { x: 25, z: 10 }, { x: -25, z: 10 },
      { x: 25, z: -10 }, { x: -25, z: -10 },
      { x: 10, z: 25 }, { x: -10, z: 25 },
      { x: 10, z: -25 }, { x: -10, z: -25 }
    ];
    positions.forEach(pos => createTree(pos.x, pos.z));
    for (let i = 15; i <= 45; i += 8) {
      createBush(i, 10);
      createBush(-i, 10);
      createBush(i, -10);
      createBush(-i, -10);
      createBush(10, i);
      createBush(-10, i);
      createBush(10, -i);
      createBush(-10, -i);
    }
  };

  const createTree = (x, z) => {
    const scene = sceneRef.current;
    const trunkGeo = new THREE.CylinderGeometry(0.3, 0.5, 2.5, 8);
    const trunkMat = new THREE.MeshStandardMaterial({ color: 0x8B4513 });
    const trunk = new THREE.Mesh(trunkGeo, trunkMat);
    trunk.position.set(x, 1.25, z);
    trunk.castShadow = true;
    scene.add(trunk);
    const leavesGeo = new THREE.ConeGeometry(2, 4, 8);
    const leavesMat = new THREE.MeshStandardMaterial({ color: 0x228B22 });
    const leaves = new THREE.Mesh(leavesGeo, leavesMat);
    leaves.position.set(x, 4, z);
    leaves.castShadow = true;
    scene.add(leaves);
    const upperGeo = new THREE.ConeGeometry(1.4, 3, 8);
    const upper = new THREE.Mesh(upperGeo, leavesMat);
    upper.position.set(x, 6, z);
    upper.castShadow = true;
    scene.add(upper);
  };

  const createBush = (x, z) => {
    const scene = sceneRef.current;
    const bushGeo = new THREE.SphereGeometry(0.7 + Math.random() * 0.5, 8, 8);
    const bushMat = new THREE.MeshStandardMaterial({ color: 0x228B22 });
    const bush = new THREE.Mesh(bushGeo, bushMat);
    bush.position.set(
      x + (Math.random() - 0.5) * 2,
      0.7,
      z + (Math.random() - 0.5) * 2
    );
    bush.castShadow = true;
    scene.add(bush);
  };

  const createVehicle = () => {
    const group = new THREE.Group();
    const type = Math.floor(Math.random() * 3);
    const colors = [0xff0000, 0x0000ff, 0x00ff00, 0xffff00, 0xffffff, 0x000000];
    const color = colors[Math.floor(Math.random() * colors.length)];
    
    // Create vehicle mesh based on type
    if (type === 0) createCar(group, color);
    else if (type === 1) createTruck(group, color);
    else createVan(group, color);
    
    // Spawn position with better lane alignment
    const startPos = Math.floor(Math.random() * 4);
    const laneOffset = Math.random() > 0.5 ? 1.5 : -1.5;  // Distinct lanes
    
    switch (startPos) {
      case 0:  // North
        group.position.set(laneOffset, 0, 40);
        group.rotation.y = Math.PI;
        break;
      case 1:  // East
        group.position.set(40, 0, laneOffset);
        group.rotation.y = -Math.PI / 2;
        break;
      case 2:  // South
        group.position.set(laneOffset, 0, -40);
        group.rotation.y = 0;
        break;
      case 3:  // West
        group.position.set(-40, 0, laneOffset);
        group.rotation.y = Math.PI / 2;
        break;
    }
    
    return group;
  };
  const checkCollisions = (vehicle, proposedPos) => {
    const minSpacing = 6.0;  // Minimum spacing between vehicles in same lane
    const intersectionBuffer = 2.0;  // Buffer zone around intersection
    
    const vehicleDir = getVehicleDirection(vehicle.rotation.y);
    
    for (const other of vehiclesRef.current) {
      if (other === vehicle) continue;
      
      const otherDir = getVehicleDirection(other.rotation.y);
      
      // Check if vehicles are in the same lane
      const inSameLane = areSameLane(proposedPos, other.position, vehicleDir);
      
      if (inSameLane && vehicleDir === otherDir) {
        // Enforce minimum spacing for vehicles in same lane
        const distance = getDirectionalDistance(proposedPos, other.position, vehicleDir);
        if (distance < minSpacing) return true;
      } else if (isNearIntersection(proposedPos) && isNearIntersection(other.position)) {
        // Only check cross-traffic near intersection
        const distance = Math.sqrt(
          Math.pow(proposedPos.x - other.position.x, 2) +
          Math.pow(proposedPos.z - other.position.z, 2)
        );
        if (distance < intersectionBuffer) return true;
      }
    }
    return false;
  };
  
  const areSameLane = (pos1, pos2, direction) => {
    if (direction === 'north' || direction === 'south') {
      return Math.abs(pos1.x - pos2.x) < 2;
    } else {  // east or west
      return Math.abs(pos1.z - pos2.z) < 2;
    }
  };
  
  const getDirectionalDistance = (pos1, pos2, direction) => {
    switch (direction) {
      case 'north':
        return Math.abs(pos1.z - pos2.z);
      case 'south':
        return Math.abs(pos1.z - pos2.z);
      case 'east':
        return Math.abs(pos1.x - pos2.x);
      case 'west':
        return Math.abs(pos1.x - pos2.x);
      default:
        return 0;
    }
  };
  
  const isNearIntersection = (pos) => {
    return Math.abs(pos.x) < 8 && Math.abs(pos.z) < 8;
  };
  const createCar = (group, color) => {
    const bodyGeo = new THREE.BoxGeometry(2, 1, 4.5);
    const bodyMat = new THREE.MeshStandardMaterial({ color, roughness: 0.2, metalness: 0.8 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.6;
    body.castShadow = true;
    group.add(body);
    const topGeo = new THREE.BoxGeometry(1.8, 0.7, 2.5);
    const top = new THREE.Mesh(topGeo, bodyMat);
    top.position.set(0, 1.45, -0.2);
    top.castShadow = true;
    group.add(top);
    const winMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.1, metalness: 0.9 });
    const windGeo = new THREE.BoxGeometry(1.7, 0.6, 0.1);
    const windshield = new THREE.Mesh(windGeo, winMat);
    windshield.position.set(0, 1.45, 1);
    group.add(windshield);
    const rearGeo = new THREE.BoxGeometry(1.7, 0.6, 0.1);
    const rearWindow = new THREE.Mesh(rearGeo, winMat);
    rearWindow.position.set(0, 1.45, -1.4);
    group.add(rearWindow);
    createWheels(group);
    createHeadlights(group);
  };

  const createTruck = (group, color) => {
    const cabGeo = new THREE.BoxGeometry(2.2, 1.8, 2);
    const cabMat = new THREE.MeshStandardMaterial({ color, roughness: 0.3, metalness: 0.7 });
    const cab = new THREE.Mesh(cabGeo, cabMat);
    cab.position.set(0, 0.9, 1.5);
    cab.castShadow = true;
    group.add(cab);
    const cargoGeo = new THREE.BoxGeometry(2.2, 2, 3.5);
    const cargoMat = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.5 });
    const cargo = new THREE.Mesh(cargoGeo, cargoMat);
    cargo.position.set(0, 1, -1);
    cargo.castShadow = true;
    group.add(cargo);
    const winMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.1, metalness: 0.9 });
    const windGeo = new THREE.BoxGeometry(2, 1, 0.1);
    const windshield = new THREE.Mesh(windGeo, winMat);
    windshield.position.set(0, 1.3, 2.55);
    group.add(windshield);
    createWheels(group, true);
    createHeadlights(group);
  };

  const createVan = (group, color) => {
    const bodyGeo = new THREE.BoxGeometry(2.2, 2.2, 5);
    const bodyMat = new THREE.MeshStandardMaterial({ color, roughness: 0.3 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.set(0, 1.1, 0);
    body.castShadow = true;
    group.add(body);
    const winMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.1, metalness: 0.9 });
    const windGeo = new THREE.BoxGeometry(2, 1, 0.1);
    const windshield = new THREE.Mesh(windGeo, winMat);
    windshield.position.set(0, 1.6, 2.55);
    group.add(windshield);
    const sideGeo = new THREE.BoxGeometry(0.1, 0.8, 2);
    const leftWindow = new THREE.Mesh(sideGeo, winMat);
    leftWindow.position.set(-1.15, 1.6, 1);
    group.add(leftWindow);
    const rightWindow = new THREE.Mesh(sideGeo, winMat);
    rightWindow.position.set(1.15, 1.6, 1);
    group.add(rightWindow);
    createWheels(group);
    createHeadlights(group);
  };

  const createWheels = (group, isTruck = false) => {
    const wheelGeo = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.7 });
    let positions;
    if (isTruck) {
      positions = [
        [-1.1, 0.4, 1.5],
        [1.1, 0.4, 1.5],
        [-1.1, 0.4, -0.5],
        [1.1, 0.4, -0.5],
        [-1.1, 0.4, -2],
        [1.1, 0.4, -2]
      ];
    } else {
      positions = [
        [-1.1, 0.4, 1.5],
        [1.1, 0.4, 1.5],
        [-1.1, 0.4, -1.5],
        [1.1, 0.4, -1.5]
      ];
    }
    positions.forEach(pos => {
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.position.set(...pos);
      wheel.rotation.z = Math.PI / 2;
      wheel.castShadow = true;
      group.add(wheel);
    });
  };

  const createHeadlights = (group) => {
    const headGeo = new THREE.CircleGeometry(0.2, 16);
    const headMat = new THREE.MeshStandardMaterial({
      color: 0xffffee,
      emissive: 0xffffee,
      emissiveIntensity: settings.timeOfDay === 'night' ? 1 : 0.2
    });
    const leftHead = new THREE.Mesh(headGeo, headMat);
    leftHead.position.set(-0.7, 0.7, 2.3);
    leftHead.rotation.y = Math.PI;
    group.add(leftHead);
    const rightHead = new THREE.Mesh(headGeo, headMat);
    rightHead.position.set(0.7, 0.7, 2.3);
    rightHead.rotation.y = Math.PI;
    group.add(rightHead);
    const tailGeo = new THREE.CircleGeometry(0.15, 16);
    const tailMat = new THREE.MeshStandardMaterial({
      color: 0xff0000,
      emissive: 0xff0000,
      emissiveIntensity: settings.timeOfDay === 'night' ? 1 : 0.3
    });
    const leftTail = new THREE.Mesh(tailGeo, tailMat);
    leftTail.position.set(-0.7, 0.7, -2.3);
    group.add(leftTail);
    const rightTail = new THREE.Mesh(tailGeo, tailMat);
    rightTail.position.set(0.7, 0.7, -2.3);
    group.add(rightTail);
  };

  const resetSimulation = () => {
    if (!sceneRef.current) return;
    vehiclesRef.current.forEach(v => sceneRef.current.remove(v));
    vehiclesRef.current = [];
    for (let i = 0; i < settings.carDensity; i++) {
      const vehicle = createVehicle();
      sceneRef.current.add(vehicle);
      vehiclesRef.current.push(vehicle);
    }
    startTrafficLightTimer();
  };

  const updateSimulation = () => {
    updateTrafficLights();
    updateVehicles();
  };

  // Updated traffic light update using activeLight (either "NS-green" or "EW-green")
  const updateTrafficLights = () => {
    if (!trafficLightsRef.current.length) return;
    
    trafficLightsRef.current.forEach(lightObj => {
      // Determine which phase this light controls
      let controlsNS = false;
      if ((lightObj.position.x < 0 && lightObj.position.z > 0) ||
          (lightObj.position.x > 0 && lightObj.position.z < 0)) {
        controlsNS = true;
      }
      
      // Get references to all relevant components
      const redLight = lightObj.getObjectByName("redLight");
      const greenLight = lightObj.getObjectByName("greenLight");
      const yellowLight = lightObj.getObjectByName("yellowLight");
      const redHalo = lightObj.getObjectByName("redHalo");
      const greenHalo = lightObj.getObjectByName("greenHalo");
      const redPointLight = lightObj.getObjectByName("redPointLight");
      const greenPointLight = lightObj.getObjectByName("greenPointLight");
      
      // EXTREME contrast between on/off states - almost turn off inactive lights entirely
      // and make active lights incredibly bright
      
      // Set default/off states
      if (redLight) redLight.material.emissiveIntensity = 0.1;
      if (greenLight) greenLight.material.emissiveIntensity = 0.1;
      if (yellowLight) yellowLight.material.emissiveIntensity = 0.1;
      if (redHalo) redHalo.visible = false;
      if (greenHalo) greenHalo.visible = false;
      if (redPointLight) redPointLight.intensity = 0;
      if (greenPointLight) greenPointLight.intensity = 0;
      
      // Set active states based on traffic direction
      if (controlsNS) {
        if (activeLight === "NS-green") {
          // North-South has green
          if (greenLight) greenLight.material.emissiveIntensity = 5.0;
          if (greenHalo) greenHalo.visible = true;
          if (greenPointLight) greenPointLight.intensity = 10;
        } else {
          // North-South has red
          if (redLight) redLight.material.emissiveIntensity = 5.0;
          if (redHalo) redHalo.visible = true;
          if (redPointLight) redPointLight.intensity = 10;
        }
      } else {
        // East-West lights
        if (activeLight === "EW-green") {
          // East-West has green
          if (greenLight) greenLight.material.emissiveIntensity = 5.0;
          if (greenHalo) greenHalo.visible = true;
          if (greenPointLight) greenPointLight.intensity = 10;
        } else {
          // East-West has red
          if (redLight) redLight.material.emissiveIntensity = 5.0;
          if (redHalo) redHalo.visible = true;
          if (redPointLight) redPointLight.intensity = 10;
        }
      }
    });
  };
  const getStatusPanelHtml = () => {
    return `
      <div class="absolute bottom-4 left-4 z-20 p-4 bg-white bg-opacity-90 rounded-lg shadow-lg">
        <h2 class="text-xl font-bold text-gray-800 mb-4">Traffic Light Status</h2>
        <div class="flex items-center space-x-4">
          <div class="flex flex-col items-center">
            <span class="text-sm uppercase font-bold mb-1">North-South</span>
            <div class="${activeLight === 'NS-green' 
              ? 'w-16 h-16 bg-green-500 rounded-full shadow-lg shadow-green-500/70 animate-pulse border-4 border-white flex items-center justify-center' 
              : 'w-16 h-16 bg-red-600 rounded-full shadow-lg shadow-red-600/70 animate-pulse border-4 border-white flex items-center justify-center'
            }">
              <span class="text-white font-bold text-lg">${activeLight === 'NS-green' ? 'GO' : 'STOP'}</span>
            </div>
          </div>
          <div class="flex flex-col items-center">
            <span class="text-sm uppercase font-bold mb-1">East-West</span>
            <div class="${activeLight === 'EW-green' 
              ? 'w-16 h-16 bg-green-500 rounded-full shadow-lg shadow-green-500/70 animate-pulse border-4 border-white flex items-center justify-center' 
              : 'w-16 h-16 bg-red-600 rounded-full shadow-lg shadow-red-600/70 animate-pulse border-4 border-white flex items-center justify-center'
            }">
              <span class="text-white font-bold text-lg">${activeLight === 'EW-green' ? 'GO' : 'STOP'}</span>
            </div>
          </div>
        </div>
        <p class="mt-4 text-sm text-gray-700">Active Vehicles: ${vehiclesRef.current?.length || 0}</p>
      </div>
    `;
  };
  

  
  const StatusPanel = () => (
    <div className="absolute bottom-4 left-4 z-20 p-4 bg-white bg-opacity-90 rounded-lg shadow-lg max-w-xs">
      <h2 className="text-lg font-semibold text-gray-700 mb-2">Traffic Status</h2>
      <div className="space-y-2">
        <p className="text-sm text-gray-700">Active Vehicles: {vehiclesRef.current?.length || 0}</p>
        <div className="flex items-center space-x-2">
          <p className="text-sm text-gray-700">Light Status:</p>
          <div className="flex items-center space-x-1">
            <div className={`w-6 h-6 rounded-full ${
              activeLight === 'NS-green' ? 'bg-green-600 shadow-lg shadow-green-400/50 animate-pulse' : 'bg-red-600 shadow-lg shadow-red-400/50 animate-pulse'
            }`}></div>
            <p className="text-sm font-bold">{activeLight === 'NS-green' ? 'North-South' : 'East-West'}</p>
          </div>
        </div>
      </div>
    </div>
  );
  

  const updateVehicles = () => {
    vehiclesRef.current.forEach(vehicle => {
      const speed = settings.carSpeed;
      const dir = getVehicleDirection(vehicle.rotation.y);
      const proposed = { x: vehicle.position.x, z: vehicle.position.z };
      if (dir === 'north') proposed.z += speed;
      else if (dir === 'east') proposed.x += speed;
      else if (dir === 'south') proposed.z -= speed;
      else if (dir === 'west') proposed.x -= speed;
      if (shouldStopAtTrafficLight(vehicle, dir)) {
        // Vehicle stops at traffic light.
      } else if (!checkCollisions(vehicle, proposed)) {
        if (dir === 'north') vehicle.position.z += speed;
        else if (dir === 'east') vehicle.position.x += speed;
        else if (dir === 'south') vehicle.position.z -= speed;
        else if (dir === 'west') vehicle.position.x -= speed;
      }
      if (
        vehicle.position.x > 50 ||
        vehicle.position.x < -50 ||
        vehicle.position.z > 50 ||
        vehicle.position.z < -50
      ) {
        sceneRef.current.remove(vehicle);
        const idx = vehiclesRef.current.indexOf(vehicle);
        if (idx > -1) {
          vehiclesRef.current.splice(idx, 1);
          const newVeh = createVehicle();
          sceneRef.current.add(newVeh);
          vehiclesRef.current.push(newVeh);
        }
      }
    });
  };

  const shouldStopAtTrafficLight = (vehicle, dir) => {
    if (lightStateRef.current === "NS-green" && dir === 'north') return false;
    if (lightStateRef.current === "NS-green" && dir === 'south') return false;
    if (lightStateRef.current === "EW-green" && dir === 'east') return false;
    if (lightStateRef.current === "EW-green" && dir === 'west') return false;
    // Otherwise, if the vehicle is approaching the intersection, force it to stop.
    const stopDist = 7;
    if (dir === 'north' && vehicle.position.z > -stopDist && vehicle.position.z < 0) return true;
    else if (dir === 'east' && vehicle.position.x > -stopDist && vehicle.position.x < 0) return true;
    else if (dir === 'south' && vehicle.position.z < stopDist && vehicle.position.z > 0) return true;
    else if (dir === 'west' && vehicle.position.x < stopDist && vehicle.position.x > 0) return true;
    return false;
  };

  const getVehicleDirection = (rotationY) => {
    const norm = ((rotationY % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    if (norm < Math.PI / 4 || norm > 7 * Math.PI / 4) return 'north';
    else if (norm >= Math.PI / 4 && norm < 3 * Math.PI / 4) return 'east';
    else if (norm >= 3 * Math.PI / 4 && norm < 5 * Math.PI / 4) return 'south';
    else return 'west';
  };

  const startTrafficLightTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      if (!settings.isAutoMode) {
        // Fallback baseline timer (alternating states) if not using auto mode.
        if (lightStateRef.current === "NS-green") {
          lightStateRef.current = "EW-green";
          setActiveLight("EW-green");
        } else {
          lightStateRef.current = "NS-green";
          setActiveLight("NS-green");
        }
      }
    }, (settings.trafficLightTiming * 1000) / 3);
  };

  const handleSettingChange = (setting, value) => {
    setSettings(prev => ({ ...prev, [setting]: value }));
    if (setting === 'timeOfDay') updateEnvironment(value);
    if (setting === 'trafficLightTiming') startTrafficLightTimer();
    if (setting === 'carDensity') resetSimulation();
  };

  return (
    <div className="relative w-screen h-screen bg-gray-100">
      {/* Three.js Simulation Canvas */}
      <div ref={mountRef} className="absolute inset-0" />

      {/* Control Panel Overlay - Top Left */}
      <div className="absolute top-4 left-4 z-20 p-6 bg-white bg-opacity-90 rounded-lg shadow-lg max-w-xs space-y-4">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Traffic Simulator</h1>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Number of Vehicles:</label>
            <div className="flex items-center space-x-2">
              <input
                type="range"
                min="1"
                max="100"
                value={settings.carDensity}
                onChange={(e) => handleSettingChange('carDensity', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg cursor-pointer"
              />
              <span className="text-gray-700 w-8 text-center">{settings.carDensity}</span>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Vehicle Speed:</label>
            <div className="flex items-center space-x-2">
              <input
                type="range"
                min="0.05"
                max="0.30"
                step="0.01"
                value={settings.carSpeed}
                onChange={(e) => handleSettingChange('carSpeed', parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg cursor-pointer"
              />
              <span className="text-gray-700 w-12 text-center">{settings.carSpeed.toFixed(2)}</span>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Traffic Light Timing (seconds):</label>
            <div className="flex items-center space-x-2">
              <input
                type="range"
                min="5"
                max="60"
                value={settings.trafficLightTiming}
                onChange={(e) => handleSettingChange('trafficLightTiming', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg cursor-pointer"
              />
              <span className="text-gray-700 w-8 text-center">{settings.trafficLightTiming}</span>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Time of Day:</label>
            <select
              value={settings.timeOfDay}
              onChange={(e) => handleSettingChange('timeOfDay', e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-2 border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="day">Day</option>
              <option value="sunset">Sunset</option>
              <option value="night">Night</option>
            </select>
          </div>
          <div className="flex items-center">
            <label className="flex items-center space-x-2 text-sm font-medium text-gray-700">
              <input
                type="checkbox"
                checked={settings.isAutoMode}
                onChange={(e) => handleSettingChange('isAutoMode', e.target.checked)}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span>Automatic Traffic Lights (RL)</span>
            </label>
          </div>
          <button
            onClick={resetSimulation}
            className="w-full px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
          >
            Reset Simulation
          </button>
        </div>
      </div>

      {/* Status Panel - Bottom Left */}
      <div className="absolute bottom-4 left-4 z-20 p-4 bg-white bg-opacity-90 rounded-lg shadow-lg max-w-xs">
        <h2 className="text-lg font-semibold text-gray-700 mb-2">Traffic Status</h2>
        <div className="space-y-2">
          <p className="text-sm text-gray-700">Active Vehicles: {vehiclesRef.current?.length || 0}</p>
          <div className="flex items-center space-x-2">
            <p className="text-sm text-gray-700">Light Status:</p>
            <div className={`w-4 h-4 rounded-full ${
              activeLight === 'NS-green' ? 'bg-green-500' :
              activeLight === 'EW-green' ? 'bg-green-500' :
              'bg-red-500'
            }`}></div>
            <p className="text-sm text-gray-700">{activeLight}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrafficSimulator;
