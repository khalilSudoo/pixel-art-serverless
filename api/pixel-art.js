import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN);

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Only POST requests allowed" });
  }

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    const { inputs } = body;

    if (!inputs) {
      return res.status(400).json({ error: "Missing 'inputs' in body" });
    }

    // Generate image
    const image = await client.textToImage({
      model: "nerijs/pixel-art-xl",
      inputs,
      parameters: {
        num_inference_steps: 5,
        guidance_scale: 7.5,
        width: 512,
        height: 512,
      },
    });

    const arrayBuffer = await image.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    res.status(200).json({ image: buffer.toString("base64") });

  } catch (error) {
    console.error("Error generating pixel art:", error);
    res.status(500).json({ error: "Failed to generate image" });
  }
}
