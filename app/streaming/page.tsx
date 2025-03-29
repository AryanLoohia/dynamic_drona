"use client";

import { useState, useEffect } from "react";
import { auth } from "@/lib/firebase";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from  "../components/Navbar"
export default function Streaming() {
  const [mounted, setMounted] = useState(false);
  const [user, setUser] = useState<any | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedImage1, setStreamedImage1] = useState<string | null>(null);
  const [streamedImage2, setStreamedImage2] = useState<string | null>(null);
  const [showModel1, setShowModel1] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const [streamedDepthImage, setStreamedDepthImage] = useState<string | null>(null);
  const [showModel, setShowModel] = useState<"model1" | "model2" | "depth">(showModel1 ? "model1" : "model2");

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      setUser(user);
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    let eventSource: EventSource | null = null;

    if (isStreaming && mounted) {
      try {
        eventSource = new EventSource("http://localhost:5001/stream");

        eventSource.onopen = () => {
          console.log("Stream connected successfully");
          setError(null);
        };

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.error) {
              console.error("Stream data error:", data.error);
              setError(data.error);
              return;
            }
            setStreamedImage1(data.model1_frame ? `data:image/jpeg;base64,${data.model1_frame}` : null);
            setStreamedImage2(data.model2_frame ? `data:image/jpeg;base64,${data.model2_frame}` : null);
            setStreamedDepthImage(data.depth_frame ? `data:image/jpeg;base64,${data.depth_frame}` : null);
          } catch (error) {
            console.error("Error parsing stream data:", error);
            setError("Failed to process stream data");
          }
        };

        eventSource.onerror = (event) => {
          console.error("Stream connection error:", event);
          setError("Stream connection failed");
          if (eventSource) {
            eventSource.close();
          }
          setIsStreaming(false);
        };
      } catch (error) {
        console.error("Error setting up stream:", error);
        setError("Failed to initialize stream");
        setIsStreaming(false);
      }
    }

    return () => {
      if (eventSource) {
        console.log("Closing stream connection");
        eventSource.close();
      }
    };
  }, [isStreaming, mounted]);

  const startStreaming = () => {
    setIsStreaming(true);
    setError(null);
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    setStreamedImage1(null);
    setStreamedImage2(null);
    setStreamedDepthImage(null);
    setError(null);
  };

  if (!mounted) return null;

  if (!user) {
    return (
      <div className="bg-zinc-900 flex flex-col items-center justify-center min-h-screen text-white">
        <h1 className="text-3xl font-bold mb-6">Login Required</h1>
        <div className="flex flex-row items-center justify-center">
        <Link href="/auth">
          <button className="px-4 mx-5 py-2 bg-red-700 rounded text-white hover:bg-red-600">
            Login
          </button>
        </Link>
        <Link href="/">
          <button className="px-4 mx-5  py-2 bg-red-700 rounded text-white hover:bg-red-600">
            Back to Home
          </button>
        </Link>
        </div>
        
      </div>
    );
  }

  return (
    <div className="bg-zinc-850 text-white min-h-screen flex flex-col items-center justify-center p-6">
      {/* Navbar */}
     <Navbar></Navbar>

      <h1 className="text-3xl font-bold mb-6 bg-slate-100 bg-clip-text ">
        Live Streaming Page
      </h1>

      {/* Streaming Section */}
      <div className="w-full max-w-4xl bg-white p-6 rounded-lg shadow-2xl mb-6">
        <h2 className="text-2xl text-[#211A1D] font-semibold mb-4">Live Video Detection</h2>

        {error && (
          <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}

        <div className="flex justify-between items-center mb-4">
          <button
            onClick={isStreaming ? stopStreaming : startStreaming}
            className={`px-4 py-2 rounded text-[#FFFFFF] w-full ${
              isStreaming ? "bg-red-600 hover:bg-red-700" : "bg-red-600 hover:bg-red-700"
            }`}
          >
            {isStreaming ? "Stop Stream" : "Start Stream"}
          </button>
        </div>

        {isStreaming && (
          <div className="w-full">
            {/* Model Selection Tabs */}
            <div className="flex space-x-4 mb-6 border-b border-gray-700">
              <button
                onClick={() => setShowModel("model1")}
                className={`pb-2 px-4 text-sm font-medium transition-colors duration-200 relative ${
                  showModel === "model1"
                    ? "text-red-400 border-b-2 border-red-400"
                    : "text-gray-400 hover:text-gray-300"
                }`}
              >
                Hazard Detection
                {showModel === "model1" && <span className="absolute bottom-0 left-0 w-full h-0.5 bg-red-400"></span>}
              </button>
              <button
                onClick={() => setShowModel("model2")}
                className={`pb-2 px-4 text-sm font-medium transition-colors duration-200 relative ${
                  showModel === "model2"
                    ? "text-red-400 border-b-2 border-red-400"
                    : "text-gray-400 hover:text-gray-300"
                }`}
              >
                Crane Defects
                {showModel === "model2" && <span className="absolute bottom-0 left-0 w-full h-0.5 bg-red-400"></span>}
              </button>
              <button
                onClick={() => setShowModel("depth")}
                className={`pb-2 px-4 text-sm font-medium transition-colors duration-200 relative ${
                  showModel === "depth"
                    ? "text-red-400 border-b-2 border-red-400"
                    : "text-gray-400 hover:text-gray-300"
                }`}
              >
                Depth Map
                {showModel === "depth" && <span className="absolute bottom-0 left-0 w-full h-0.5 bg-red-400"></span>}
              </button>
            </div>

            {/* Live Streaming Display */}
            <div className="rounded-lg overflow-hidden shadow-lg">
              {showModel === "model1" && streamedImage1 && (
                <img src={streamedImage1} alt="Hazard Detection" className="w-full" />
              )}
              {showModel === "model2" && streamedImage2 && (
                <img src={streamedImage2} alt="Crane Defects" className="w-full" />
              )}
              {showModel === "depth" && streamedDepthImage && (
                <img src={streamedDepthImage} alt="Depth Map" className="w-full" />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}