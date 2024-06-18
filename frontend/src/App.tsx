import { useEffect, useRef, useState } from "react";
import { ReactSketchCanvas } from "react-sketch-canvas";
import "./App.css";

function App() {
  const signatureCanvasRef = useRef<any>(null);

  const [result, setResult] = useState<{
    prediction: string;
    image: string;
  } | null>(null);

  const getBase64AsFile = async (
    imageURL: string,
    name: string,
    type: string
  ): Promise<File> => {
    const data = await fetch(imageURL);
    const blob = await data.blob();
    return new File([blob], name, { type });
  };

  const clearImage = () => {
    signatureCanvasRef.current?.clearCanvas();
    setResult(null);
  };

  const sendImage = async () => {
    if (!signatureCanvasRef.current) return;
    const image = await signatureCanvasRef.current.exportImage("png");
    console.log(image);

    if (!image) return;
    const file = await getBase64AsFile(image, "signature", "image/png");

    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
      headers: {
        Accept: "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)",
      },
    });

    const data = await response.json();
    setResult(data);
  };

  useEffect(() => {}, []);

  return (
    <>
      <h2>A Palavra é: {result?.prediction ?? "..."}</h2>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <ReactSketchCanvas
          width="800px"
          height="300px"
          strokeWidth={25}
          strokeColor="black"
          ref={signatureCanvasRef}
        />
        <div
          style={{
            width: "100%",
            alignItems: "center",
            display: "flex",
            justifyContent: "space-between",
            flexDirection: "row",
            gap: 8,
            paddingTop: 8,
          }}
        >
          {result?.image ? (
            <img
              width={600}
              height={250}
              src={`data:image/png;base64,${result.image}`}
            />
          ) : (
            <div
              style={{ width: 600, height: 250, backgroundColor: "white" }}
            ></div>
          )}
          <div
            style={{
              width: "30%",
              alignItems: "flex-start",
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            <button
              style={{ backgroundColor: "#5f0096", width: "100%", height: 64 }}
              onClick={sendImage}
            >
              Enviar
            </button>
            <button
              style={{
                backgroundColor: "white",
                width: "100%",
                color: "#202020",
              }}
              onClick={clearImage}
            >
              Apagar
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
