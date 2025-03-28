"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { auth } from "@/lib/firebase";
import { signInWithEmailAndPassword, createUserWithEmailAndPassword } from "firebase/auth";
import Link from "next/link"

import React, { useEffect } from "react";

import {
  useMotionTemplate,
  useMotionValue,
  motion,
  animate,
} from "framer-motion";

const COLORS_TOP = ["#FF4C4C", "#D72638", "#E63946", "#FF6B6B"];
export default function AuthPage() {
    const color = useMotionValue(COLORS_TOP[0]);

    useEffect(() => {
      animate(color, COLORS_TOP, {
        ease: "easeInOut",
        duration: 10,
        repeat: Infinity,
        repeatType: "mirror",
      });
    }, []);
  
    const backgroundImage = useMotionTemplate`radial-gradient(125% 125% at 50% 0%, #020617 50%, ${color})`;
  const [isSignup, setIsSignup] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (isSignup) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
      router.push("/dashboard"); // Redirect after successful login/signup
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.section
    style={{
      backgroundImage,
    }}
    className="relative grid min-h-screen place-content-center overflow-hidden bg-gray-950 px-4 py-24 text-gray-200"
  >
    {/* <div className="flex flex-col items-center justify-center min-h-screen bg-zinc-900"> */}
      <div className="p-6 bg-white flex flex-col rounded shadow-md w-[80vw] h-[60vh]">
        <h2 className="text-[3rem] font-semibold text-center text-red-500">
          {isSignup ? "Sign Up" : "Login"}
        </h2>

        {error && <p className="text-red-500 text-sm text-center">{error}</p>}

        <form onSubmit={handleAuth} className="mt-4">
          <input
            type="email"
            placeholder="Email"
            className="mt-2 p-2 border rounded w-full text-red-500"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            className="mt-2 p-2 border rounded w-full text-red-500"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button
            type="submit"
            className="mt-4 w-full bg-red-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
            disabled={loading}
          >
            {loading ? "Processing..." : isSignup ? "Sign Up" : "Login"}
          </button>
        </form>

        <p className="text-center mt-4 text-black text-sm">
          {isSignup ? "Already have an account?" : "Don't have an account?"}
          <button
            className="text-red-500 ml-1 underline"
            onClick={() => setIsSignup(!isSignup)}
          >
            {isSignup ? "Login" : "Sign Up"}
          </button>
        </p>
        <div className="flex justify-center">
  <img
    className="relative pt-5 h-[25vh] w-[50vw] sm:w-[40vw] md:w-[30vw] lg:w-[20vw] max-w-full object-contain"
    src="/logo.jpeg"
    alt="Logo"
  />
</div>
      </div>
      <div className="pt-4">
        <Link href="/">
        <button type="button" className="text-white bg-red-700 hover:bg-red-800 focus:ring-4 focus:outline-none focus:ring-red-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center inline-flex items-center dark:bg-red-600 dark:hover:bg-red-700 dark:focus:ring-red-800">
Back to Home
<svg className="rtl:rotate-180 w-3.5 h-3.5 ms-2" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 10">
<path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M1 5h12m0 0L9 1m4 4L9 9"/>
</svg>
</button>
        </Link>
      </div>
    
    </motion.section>
  );
}
