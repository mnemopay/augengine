/**
 * Remote API Provider for connecting the AugEngine engine to powerful
 * external models (OpenAI, vLLM, etc.) for high-reasoning characters.
 */
export class RemoteApiProvider {
  private endpoint: string;
  private apiKey: string;
  private model: string;

  constructor(endpoint: string, apiKey: string, model = "gpt-4-turbo") {
    this.endpoint = endpoint;
    this.apiKey = apiKey;
    this.model = model;
  }

  async generate(prompt: string): Promise<string> {
    try {
      const response = await fetch(this.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model: this.model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.7,
        })
      });

      if (!response.ok) {
        throw new Error(`Remote API error: ${response.status}`);
      }

      const data = await response.json();
      return data.choices[0].message.content;
    } catch (e) {
      console.warn("[RemoteApiProvider] Inference failed:", e);
      return `{"action": "idle"}`;
    }
  }
}
