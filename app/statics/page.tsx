"use client";

import { useState, useEffect } from "react";
import { auth } from "@/lib/firebase";
import { useRouter } from "next/navigation";
import {
  User,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
} from "firebase/auth";
import Link from "next/link";
import Navbar from "../components/Navbar";
export default function Statics(){
    const [mounted, setMounted] = useState(false);
    const [user, setUser] = useState<User | null>(null);
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [model1ImagePath, setModel1ImagePath] = useState<string | undefined>(undefined);
    const [model2ImagePath, setModel2ImagePath] = useState<string | undefined>(undefined);
    const [model1VideoPath, setModel1VideoPath] = useState<string | undefined>(undefined);
    const [model2VideoPath, setModel2VideoPath] = useState<string | undefined>(undefined);
    const [showModel1, setShowModel1] = useState(true);
   
    const [error, setError] = useState("");
   
    const router = useRouter();
  
    // Handle client-side mounting
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

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0]);
    }
  };

  const handleUpload = async (file: File) => {
    setLoading(true);
    setModel1ImagePath(undefined);
    setModel2ImagePath(undefined);
    setModel1VideoPath(undefined);
    setModel2VideoPath(undefined);
  
    try {
      const formData = new FormData();
      formData.append('file', file);
  
      const response = await fetch('http://localhost:5001/upload', {
        method: 'POST',
        body: formData,
      });
  
      const data = await response.json();
  
      if (!data.success) {
        throw new Error(data.error || 'Failed to process file');
      }
  
      let queryParams = "";
  
      // Redirect based on uploaded file type
      if (data.model1_image_path && data.model2_image_path) {
        queryParams = `?model1Image=${data.model1_image_path}&model2Image=${data.model2_image_path}`;
      } else if (data.model1_video_path && data.model2_video_path) {
        queryParams = `?model1Video=${data.model1_video_path}&model2Video=${data.model2_video_path}`;
      }
  
      // Redirect to results page with query params
      router.push(`/results${queryParams}`);
  
    } catch (error) {
      console.error('Error:', error);
      alert(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

   // Don't render anything until mounted
   if (!mounted) {
    return null;
  }
  
  

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

  return(
    <div className="bg-zinc-850 text-white min-h-screen flex flex-col items-center justify-center p-6">
   
   <Navbar></Navbar>
   <h1 className="text-5xl font-bold mb-6 bg-gray-100 bg-clip-text text-transparent">
        Object Detection Dashboard
      </h1>

      <div className="w-full max-w-4xl bg-zinc-800 p-6 rounded-lg shadow-2xl mb-6">
        <h2 className="text-2xl text-center text-[#FFFFFF] font-semibold mb-4">Upload Image/Video for Detection</h2>
        
         <div className="flex flex-col pb-5">
      
      <input
        type="file"
        id="fileInput"
        onChange={handleFileChange}
        accept="image/*,video/*"
        className="hidden"
      />

      
      <label
        htmlFor="fileInput"
        className="cursor-pointer w-[10rem] bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4  shadow-md transition-all items-center text-center justify-center"
      >
        Choose File
      </label>

      
      {file && (
        <p className="mt-2 text-sm text-gray-300">{file.name}</p>
      )}
    </div>
        <button 
          onClick={() => file && handleUpload(file)} 
          className="px-4 py-2 bg-red-600 rounded text-[#ffffff] hover:bg-red-700 w-full"
          disabled={loading || !file}
        >
          {loading ? "Processing..." : "Upload"}
        </button>
      </div>
   </div>
  )

}