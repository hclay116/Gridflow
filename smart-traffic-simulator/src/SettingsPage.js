import React, { useState } from 'react';
import { Play, Car, Users, Gauge, Calendar } from 'lucide-react';

const SettingsPage = ({ onStartSimulation }) => {
  const [settings, setSettings] = useState({
    carDensity: 8,
    pedestrianDensity: 5,
    trafficLightTiming: 20,
    isAutoMode: true,
    carSpeed: 0.15,
    pedestrianSpeed: 0.08,
    timeOfDay: 'day',
  });

  const handleSettingChange = (setting, value) => {
    setSettings(prev => ({ ...prev, [setting]: value }));
  };

  const handleStart = () => {
    onStartSimulation(settings);
  };

  // Helper function to get background and text colors based on time of day
  const getTimeOfDayStyles = () => {
    switch(settings.timeOfDay) {
      case 'day':
        return { bgClass: 'from-blue-500 to-indigo-600', textClass: 'text-white' };
      case 'sunset':
        return { bgClass: 'from-orange-400 to-pink-600', textClass: 'text-white' };
      case 'night':
        return { bgClass: 'from-indigo-900 to-purple-900', textClass: 'text-white' };
      default:
        return { bgClass: 'from-blue-500 to-indigo-600', textClass: 'text-white' };
    }
  };

  const { bgClass, textClass } = getTimeOfDayStyles();

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4 sm:p-8">
      <div className="w-full max-w-4xl">
        {/* Header with dynamic background based on time of day */}
        <div className={`bg-gradient-to-r ${bgClass} rounded-t-2xl p-8 shadow-lg`}>
          <h1 className={`text-4xl font-extrabold mb-2 ${textClass} text-center tracking-tight`}>
            Traffic Simulator
          </h1>
          <p className={`${textClass} text-center text-lg opacity-90`}>
            Fine-tune your simulation parameters
          </p>
        </div>
        
        {/* Main content area */}
        <div className="bg-white rounded-b-2xl shadow-xl p-8 border border-gray-100">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Traffic Density Section */}
            <div className="bg-gradient-to-br from-slate-50 to-blue-50 p-6 rounded-xl shadow-sm border border-blue-100 hover:shadow-md transition-all">
              <div className="flex items-center mb-4">
                <div className="bg-blue-100 p-2 rounded-lg mr-3">
                  <Car size={20} className="text-blue-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Traffic Density</h2>
              </div>
              
              <div className="space-y-6">
                {[{ label: 'Number of Vehicles', key: 'carDensity', min: 1, max: 20, color: 'blue' },
                  { label: 'Number of Pedestrians', key: 'pedestrianDensity', min: 1, max: 20, color: 'indigo' }].map(({ label, key, min, max, color }) => (
                    <div key={key}>
                      <div className="flex justify-between items-center mb-2">
                        <label className="text-sm font-medium text-gray-700">{label}</label>
                        <span className={`text-sm bg-${color}-100 text-${color}-800 font-medium py-1 px-2 rounded-full`}>
                          {settings[key]}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={min}
                        max={max}
                        value={settings[key]}
                        onChange={(e) => handleSettingChange(key, parseInt(e.target.value))}
                        className={`w-full h-2 bg-${color}-100 rounded-lg appearance-none cursor-pointer`}
                        style={{ 
                          backgroundImage: `linear-gradient(to right, ${color === 'blue' ? '#3b82f6' : '#6366f1'} 0%, ${color === 'blue' ? '#3b82f6' : '#6366f1'} ${(settings[key] - min) / (max - min) * 100}%, #e2e8f0 ${(settings[key] - min) / (max - min) * 100}%)` 
                        }}
                      />
                    </div>
                ))}
              </div>
            </div>

            {/* Speed Controls Section */}
            <div className="bg-gradient-to-br from-slate-50 to-green-50 p-6 rounded-xl shadow-sm border border-green-100 hover:shadow-md transition-all">
              <div className="flex items-center mb-4">
                <div className="bg-green-100 p-2 rounded-lg mr-3">
                  <Gauge size={20} className="text-green-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Speed Controls</h2>
              </div>
              
              <div className="space-y-6">
                {[{ label: 'Vehicle Speed', key: 'carSpeed', min: 0.05, max: 0.30, step: 0.01, color: 'green' },
                  { label: 'Pedestrian Speed', key: 'pedestrianSpeed', min: 0.02, max: 0.15, step: 0.01, color: 'emerald' }].map(({ label, key, min, max, step, color }) => (
                    <div key={key}>
                      <div className="flex justify-between items-center mb-2">
                        <label className="text-sm font-medium text-gray-700">{label}</label>
                        <span className={`text-sm bg-${color}-100 text-${color}-800 font-medium py-1 px-2 rounded-full`}>
                          {settings[key].toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={min}
                        max={max}
                        step={step}
                        value={settings[key]}
                        onChange={(e) => handleSettingChange(key, parseFloat(e.target.value))}
                        className={`w-full h-2 bg-${color}-100 rounded-lg appearance-none cursor-pointer`}
                        style={{ 
                          backgroundImage: `linear-gradient(to right, ${color === 'green' ? '#22c55e' : '#10b981'} 0%, ${color === 'green' ? '#22c55e' : '#10b981'} ${(settings[key] - min) / (max - min) * 100}%, #e2e8f0 ${(settings[key] - min) / (max - min) * 100}%)` 
                        }}
                      />
                    </div>
                ))}
              </div>
            </div>

            {/* Environment Settings */}
            <div className="md:col-span-2 bg-gradient-to-br from-slate-50 to-purple-50 p-6 rounded-xl shadow-sm border border-purple-100 hover:shadow-md transition-all">
              <div className="flex items-center mb-4">
                <div className="bg-purple-100 p-2 rounded-lg mr-3">
                  <Calendar size={20} className="text-purple-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Environment</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-3">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Time of Day</label>
                  <div className="grid grid-cols-3 gap-3">
                    {['day', 'sunset', 'night'].map((time) => (
                      <button
                        key={time}
                        onClick={() => handleSettingChange('timeOfDay', time)}
                        className={`flex flex-col items-center justify-center p-4 rounded-xl transition-all ${
                          settings.timeOfDay === time 
                            ? 'bg-purple-100 border-2 border-purple-500 shadow-md' 
                            : 'bg-white border border-gray-200 hover:border-purple-300'
                        }`}
                      >
                        <div className={`w-12 h-12 rounded-full mb-2 ${
                          time === 'day' ? 'bg-yellow-400' : 
                          time === 'sunset' ? 'bg-gradient-to-br from-orange-400 to-pink-500' : 
                          'bg-indigo-900'
                        } flex items-center justify-center`}>
                          {time === 'day' && <span className="text-lg">☀️</span>}
                          {time === 'sunset' && <span className="text-lg">🌅</span>}
                          {time === 'night' && <span className="text-lg">🌙</span>}
                        </div>
                        <span className={`text-sm font-medium ${settings.timeOfDay === time ? 'text-purple-800' : 'text-gray-700'}`}>
                          {time.charAt(0).toUpperCase() + time.slice(1)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Start Button */}
          <button
            onClick={handleStart}
            className="w-full mt-8 px-6 py-4 bg-gradient-to-r from-blue-600 to-indigo-700 text-white rounded-xl shadow-lg 
                      hover:from-blue-700 hover:to-indigo-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 
                      flex items-center justify-center gap-3 text-lg font-semibold transition-all transform hover:scale-[1.02]"
          >
            <Play size={24} className="animate-pulse" />
            Start Simulation
          </button>
        </div>

        {/* Footer */}
        <div className="mt-4 text-center text-sm text-slate-500">
          Adjust the settings above to customize your traffic simulation experience
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;