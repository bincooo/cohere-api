package cohere

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/bincooo/emit.io"
	"github.com/sirupsen/logrus"
	"io"
	"net/http"
)

const (
	baseUrl = "https://api.cohere.com"

	COMMAND               = "command"
	COMMAND_LIGHT         = "command-light"
	COMMAND_LIGHT_NIGHTLY = "command-light-nightly"
	COMMAND_NIGHTLY       = "command-nightly"
	COMMAND_R             = "command-r"
	COMMAND_R_PLUS        = "command-r-plus"
	COMMAND_R_202408      = "command-r-08-2024"
	COMMAND_R_PLUS_202408 = "command-r-plus-08-2024"
)

type block struct {
	Finished  bool          `json:"is_finished"`
	Event     string        `json:"event_type"`
	Id        string        `json:"generation_id"`
	Text      string        `json:"text"`
	Reason    string        `json:"finish_reason"`
	ToolCalls []interface{} `json:"tool_calls"`
}

type Message struct {
	Role    string
	Message string
}

type ToolObject struct {
	Tools   []ToolCall
	Results []ToolResult
}

type ToolCall struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Param       map[string]interface{} `json:"parameter_definitions"`
}

type ToolResult struct {
	Call    interface{}   `json:"call"`
	Outputs []interface{} `json:"outputs"`
}

type Chat struct {
	temperature   float32
	maxTokens     int
	model         string
	seed          int
	token         string
	proxies       string
	isChat        bool
	stopSequences []string
	topK          int
	safety        string
	client        *emit.Session
}

func (c *Chat) Proxies(proxies string) {
	c.proxies = proxies
}

func (c *Chat) MaxTokens(maxTokens int) {
	if maxTokens < 1 {
		return
	}
	c.maxTokens = maxTokens
}

func (c *Chat) TopK(topK int) {
	if topK < 1 {
		return
	}
	c.topK = topK
}

func (c *Chat) Seed(seed int) {
	if seed < 1 {
		return
	}
	c.seed = seed
}

func (c *Chat) Safety(safety string) {
	c.safety = safety
}

func (c *Chat) StopSequences(stopSequences []string) {
	if len(stopSequences) == 0 {
		return
	}
	c.stopSequences = stopSequences
}

func (c *Chat) Client(client *emit.Session) {
	c.client = client
}

func New(token string, temperature float32, model string, isChat bool) Chat {
	return Chat{
		token:         token,
		temperature:   temperature,
		model:         model,
		seed:          -1,
		topK:          40,
		maxTokens:     4096,
		isChat:        isChat,
		stopSequences: make([]string, 0),
	}
}

func (c *Chat) Reply(ctx context.Context, pMessages []Message, system, message string, toolObject ToolObject) (ch chan string, err error) {
	var pathname = "/v1/chat"
	var response *http.Response
	payload := c.makePayload(pMessages, system, message, c.isChat, toolObject)
	if !c.isChat {
		pathname = "/v1/generate"
	}

	response, err = emit.ClientBuilder(c.client).
		Proxies(c.proxies).
		Context(ctx).
		URL(baseUrl+pathname).
		Method(http.MethodPost).
		Header("Authorization", "Bearer "+c.token).
		Header("Accept-Language", "en-US,en;q=0.9").
		Header("Origin", "https://dashboard.cohere.com").
		Header("Referer", "https://dashboard.cohere.com/").
		JHeader().
		Body(payload).
		DoC(emit.Status(http.StatusOK), emit.IsSTREAM)
	if err != nil {
		return nil, err
	}

	ch = make(chan string)
	go resolve(ch, response)
	return ch, nil
}

func (c *Chat) makePayload(pMessages []Message, system, message string, isChat bool, toolObject ToolObject) (payload map[string]interface{}) {
	if c.temperature < 0 {
		c.temperature = 0.95
	}

	if isChat {
		payload = map[string]interface{}{
			"chat_history":      pMessages,
			"connectors":        make([]string, 0),
			"message":           message,
			"model":             c.model,
			"preamble":          system,
			"prompt_truncation": "OFF",
			"stream":            true,
			"temperature":       c.temperature,
			"tools":             toolObject.Tools,
			"tool_results":      toolObject.Results,
		}

		if c.seed > 0 {
			payload["seed"] = c.seed
		}

		if c.safety != "" {
			payload["safety_mode"] = c.safety
		}

	} else {
		payload = map[string]interface{}{
			"k":             c.topK,
			"model":         c.model,
			"max_tokens":    c.maxTokens,
			"prompt":        message,
			"raw_prompting": false,
			"stream":        true,
			"temperature":   c.temperature,
		}
	}

	if len(c.stopSequences) > 0 {
		payload["stop_sequences"] = c.stopSequences
	}
	return payload
}

func MergeMessages(messages []map[string]string) string {
	if len(messages) == 0 {
		return ""
	}

	lastRole := ""
	buf := new(bytes.Buffer)

	for _, message := range messages {
		if lastRole == "" || lastRole != message["role"] {
			lastRole = message["role"]
			buf.WriteString(fmt.Sprintf("\n%s: %s", message["role"], message["content"]))
			continue
		}
		buf.WriteString(fmt.Sprintf("\n%s", message["content"]))
	}

	return buf.String()
}

func resolve(ch chan string, response *http.Response) {
	defer close(ch)
	defer response.Body.Close()

	buf := new(bytes.Buffer)
	r := bufio.NewReader(response.Body)
	for {
		line, prefix, err := r.ReadLine()
		buf.Write(line)

		if err != nil {
			if err != io.EOF {
				ch <- fmt.Sprintf("error: %v", err)
			}
			if str := buf.String(); len(str) > 0 {
				ch <- "text: " + str
			}
			return
		}
		if prefix {
			continue
		}

		logrus.Tracef("--------- ORIGINAL MESSAGE ---------")
		logrus.Tracef("%s", buf.Bytes())

		var b block
		if err = json.Unmarshal(buf.Bytes(), &b); err != nil {
			ch <- fmt.Sprintf("error: %v", err)
			return
		}
		buf.Reset()

		if b.Finished {
			return
		}

		if b.Event == "text-generation" {
			ch <- "text: " + b.Text
			continue
		}

		if b.Event == "tool-calls-generation" {
			marshal, e := json.Marshal(b.ToolCalls)
			if e != nil {
				ch <- fmt.Sprintf("error: %v", e)
				return
			}
			ch <- fmt.Sprintf("tool: %s", marshal)
		}
	}
}
