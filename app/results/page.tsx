"use client";
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import Navbar from "../components/Navbar";

export default function ResultsPage() {
  const searchParams = useSearchParams();
  const model1Image = searchParams.get("model1Image");
  const model2Image = searchParams.get("model2Image");
  const model1Video = searchParams.get("model1Video");
  const model2Video = searchParams.get("model2Video");

  // State to manage active tab
  const [activeTab, setActiveTab] = useState<"hazard" | "crane">("hazard");

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
        </div>

        {/* Tab Content */}
        <div className="w-full bg-zinc-800 p-6 border border-gray-700 rounded-lg shadow-lg">
          {activeTab === "hazard" && model1Image && (
            <div>
              <h3 className="text-xl mb-4 text-white-400 font-medium">Hazard Detection</h3>
              <div className="rounded-lg overflow-hidden shadow-lg mb-6">
                <img src={`http://localhost:5001/runs/${model1Image}`} alt="Hazard Detection" className="w-full" />
              </div>
              {model1Video && (
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

          {activeTab === "crane" && model2Image && (
            <div>
              <h3 className="text-xl mb-4 text-white-400 font-medium">Crane Defects</h3>
              <div className="rounded-lg overflow-hidden shadow-lg mb-6">
                <img src={`http://localhost:5001/runs/${model2Image}`} alt="Crane Defects" className="w-full" />
              </div>
              {model2Video && (
                <div>
                  <h3 className="text-xl mb-4 text-purple-400 font-medium">Crane Defects Video</h3>
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
