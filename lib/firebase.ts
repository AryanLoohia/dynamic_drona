import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
    apiKey: "AIzaSyBmvG39KSkx-G226eMSKL67AWYfaCdwJFI",
    authDomain: "dynamic-938e2.firebaseapp.com",
    projectId: "dynamic-938e2",
    storageBucket: "dynamic-938e2.firebasestorage.app",
    messagingSenderId: "942372957749",
    appId: "1:942372957749:web:2254b7c9c0048d6512cb7c",
    measurementId: "G-EY407KGFP5"
}

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

// Initialize analytics only on the client side
let analytics;
if (typeof window !== 'undefined') {
    analytics = getAnalytics(app);
}