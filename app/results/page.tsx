"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import Navbar from "../components/Navbar";
import './page.css'; // Import your CSS file for the animated line

export default function ResultsPage() {
  const searchParams = useSearchParams();
  const model1Image = searchParams.get("model1Image");
  const model2Image = searchParams.get("model2Image");
  const model1Video = searchParams.get("model1Video");
  const model2Video = searchParams.get("model2Video");
  const depthImage = searchParams.get("depthImage");
  const depthVideo = searchParams.get("depthVideo");

  // State to store parsed object descriptions
  const [objectDescriptions, setObjectDescriptions] = useState<Record<string, string>>({});

  useEffect(() => {
    // Parse the objectDescriptions from URL parameter
    const objectDescriptionsParam = searchParams.get("objectDescriptions");
    console.log('Raw objectDescriptions param:', objectDescriptionsParam);

    if (objectDescriptionsParam) {
      try {
        const decoded = decodeURIComponent(objectDescriptionsParam);
        console.log('Decoded param:', decoded);
        
        const parsed = JSON.parse(decoded);
        console.log('Parsed objectDescriptions:', parsed);
        
        setObjectDescriptions(parsed);
      } catch (error) {
        console.error('Error parsing objectDescriptions:', error);
        setObjectDescriptions({});
      }
    } else {
      console.log('No objectDescriptions parameter found');
    }
  }, [searchParams]);

  // Add this log to see what's being rendered
  console.log('Current objectDescriptions state:', objectDescriptions);

  // State to manage active tab
  const [activeTab, setActiveTab] = useState<"hazard" | "crane" | "depth">("hazard");

  return (
    <div className="bg-zinc-900 text-white min-h-screen flex flex-col items-center justify-center p-6">
      <Navbar />

      <div className="pt-16 w-full max-w-4xl">
        <h1 className="text-3xl font-bold mb-6 text-center">Detection Results</h1>

        {/* Tab Buttons */}
        <div className="flex justify-center mb-6 border-b border-gray-700">
          <button
            className={`px-6 py-2 font-medium ${activeTab === "hazard" ? "border-b-2 border-red-500 text-white-400" : "text-gray-400"}`}
            onClick={() => setActiveTab("hazard")}
          >
            Hazard Detection
          </button>
          <button
            className={`px-6 py-2 font-medium ${activeTab === "crane" ? "border-b-2 border-red-500 text-white-400" : "text-gray-400"}`}
            onClick={() => setActiveTab("crane")}
          >
            Crane Defects
          </button>
          <button
            className={`px-6 py-2 font-medium ${activeTab === "depth" ? "border-b-2 border-red-500 text-white-400" : "text-gray-400"}`}
            onClick={() => setActiveTab("depth")}
          >
            Depth Map
          </button>
        </div>

        {/* Tab Content */}
        <div className="w-full bg-zinc-800 p-6 border border-gray-700 rounded-lg shadow-lg">
          {activeTab === "hazard" && (
            <div>
              {/* <div className="h-[5rem] w-[5rem]">{ model1Image !== undefined}</div> */}
              {model1Image !== "undefined" && (
                <div>
                  <h3 className="text-xl mb-4 text-white-400 font-medium">Hazard Detection</h3>
                  
                  <div className="rounded-lg overflow-hidden shadow-lg mb-6">
                    <img src={`http://localhost:5001/runs/${model1Image}`} alt="Hazard Detection" className="w-full" ></img>
                  </div>  
                </div>
              )}
              {model1Video !== "undefined" && (
                <div>
                  <h3 className="text-xl mb-4 text-white-400 font-medium">Hazard Detection Video</h3>
                  <div className="rounded-lg overflow-hidden shadow-lg">
                    <video controls className="w-full">
                      <source src={`http://localhost:5001/runs/${model1Video}`} type="video/mp4" />
                      Your browser does not support video playback.
                    </video>
                  </div>
                </div>
              )}
            </div>
          )}
          {activeTab === "crane" && (
            <div>
              {model2Image !== "undefined" && (
                <div>
                  <h3 className="text-xl mb-4 text-white-400 font-medium">Crane Defects</h3>
                  <div className="rounded-lg overflow-hidden shadow-lg mb-6">
                    <img src={`http://localhost:5001/runs/${model2Image}`} alt="Crane Defects" className="w-full" />
                  </div>
                </div>
              )}
              {model2Video !== "undefined" && (
                <div>
                  <h3 className="text-xl mb-4 text-white-400 font-medium">Crane Defects Video</h3>
                  <div className="rounded-lg overflow-hidden shadow-lg">
                    <video controls className="w-full">
                      <source src={`http://localhost:5001/runs/${model2Video}`} type="video/mp4" />
                      Your browser does not support video playback.
                    </video>
                  </div>
                </div>
              )}
            </div>
          )}
          {activeTab === "depth" && (
            <div>
              {/* <div className="h-[5rem] w-[5rem]">{ model1Image !== undefined}</div> */}
              {depthImage !== "undefined" && (
                <div>
                  <h3 className="text-xl mb-4 text-white-400 font-medium">Depth Map</h3>
                  <div className="rounded-lg overflow-hidden shadow-lg mb-6">
                    <img src={`http://localhost:5001/runs/${depthImage}`} alt="Depth Map" className="w-full" ></img>
                  </div>  
                </div>
              )}
              {depthVideo !== "undefined" && (
                <div>
                  <h3 className="text-xl mb-4 text-white-400 font-medium">Depth Map Video</h3>
                  <div className="rounded-lg overflow-hidden shadow-lg">
                    <video controls className="w-full">
                      <source src={`http://localhost:5001/runs/${depthVideo}`} type="video/mp4" />
                      Your browser does not support video playback.
                    </video>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Animated Line */}
        <div className="animated-line my-6"></div>

        {/* Add LLM Output Table */}
        <div className="w-full bg-zinc-800 p-6 border border-gray-700 rounded-lg shadow-lg">
          <div className="mb-6">
            <h3 className="text-xl mb-4 text-white-400 font-medium">Detection Analysis</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full bg-zinc-700 rounded-lg">
                <thead>
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider border-b border-zinc-600">
                      Detected Object
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider border-b border-zinc-600">
                      Analysis
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-600">
                  {Object.keys(objectDescriptions).length === 0 ? (
                    <tr>
                      <td colSpan={2} className="px-6 py-4 text-center text-sm text-gray-200">
                        No detections available
                      </td>
                    </tr>
                  ) : (
                    Object.entries(objectDescriptions).map(([object, description], index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-200">
                          {object}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-200">
                          {description}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Back Button */}
        <div className="mt-6 flex justify-center">
          <Link href="/statics">
            <button className="bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-6 rounded-lg shadow-md transition-all">
              Predict Again
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}
