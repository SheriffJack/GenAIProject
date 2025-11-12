import axios from "axios";

const API_BASE_URL = "http://localhost:5000";

export const analyzeText = async (text) => {
  const response = await axios.post(`${API_BASE_URL}/analyze`, { text });
  return response.data;
};
